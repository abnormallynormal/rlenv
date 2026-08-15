import os, random, subprocess, sys, threading
import tkinter as tk
from collections import deque
from pathlib import Path
from queue import Empty, SimpleQueue
from tkinter import messagebox, ttk

import torch

from arcade.agents import DQNPolicy, RandomAgent
from arcade.environments import ENVIRONMENTS
from arcade.resources import resource_path
from value.agent.buffer import Buffer
from value.agent.dqn import Agent
from value.train import train_loop
from value.modules.flappy.telemetry import TelemetryRecorder as FlappyTelemetry
from value.modules.snake.telemetry import TelemetryRecorder as SnakeTelemetry


COLORS = {"background":"#11151c", "panel":"#1b2430", "panel_light":"#263241",
          "text":"#f4f7fb", "muted":"#9eabc0", "accent":"#52e0a3",
          "danger":"#ff657a", "sky":"#7bc8f6"}
TRAINING = {"Flappy Bird": (0.999, FlappyTelemetry), "Snake": (0.995, SnakeTelemetry)}


class Runner:
    """Keeps each comparison environment's use of the global RNG independent."""
    def __init__(self, env, agent, seed):
        self.env, self.agent, self.seed = env, agent, seed
        random.seed(seed); self.random_state = random.getstate()
        self.reward = 0.; self.episodes = []; self.reset()

    def _call(self, function):
        previous = random.getstate(); random.setstate(self.random_state)
        result = function(); self.random_state = random.getstate(); random.setstate(previous)
        return result

    def reset(self):
        self.state = self._call(self.env.reset); self.reward = 0.

    def step(self):
        action = self.agent.select_action(self.state)
        self.state, reward, done = self._call(lambda: self.env.step(action))
        self.reward += reward
        if done:
            self.episodes.append((self.env.score, self.reward)); self.episodes = self.episodes[-20:]
            self.reset()


class RLArcade(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL Arcade"); self.geometry("1120x740"); self.minsize(940, 640)
        self.configure(bg=COLORS["background"]); self.protocol("WM_DELETE_WINDOW", self.close)
        self.environment_name = tk.StringVar(value="Flappy Bird")
        self.mode = tk.StringVar(value="Comparison")
        self.speed = tk.IntVar(value=2); self.paused = False
        self.runners = []; self.stop_event = threading.Event(); self.worker = None
        self.playback_queue = deque(); self.playback = deque(); self.progress = SimpleQueue()
        self.renderer_ready = {"ready": True}; self.telemetry_lock = threading.Lock()
        self.training_stats = {"episode": 0, "epsilon": .9999, "score": 0, "reward": 0.}; self.training_scores = []
        self._style(); self._ui(); self.load_mode(); self.after(35, self.tick)

    def _style(self):
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI",10))
        style.configure("Title.TLabel", background=COLORS["background"], foreground=COLORS["text"], font=("Segoe UI Semibold",24))
        style.configure("Sub.TLabel", background=COLORS["background"], foreground=COLORS["muted"])
        style.configure("Metric.TLabel", background=COLORS["panel"], foreground=COLORS["accent"], font=("Segoe UI Semibold",16))
        style.configure("TButton", font=("Segoe UI Semibold",10), padding=(10,7), background=COLORS["panel_light"], foreground=COLORS["text"])

    def _ui(self):
        header = ttk.Frame(self); header.pack(fill="x", padx=24, pady=(18,12))
        ttk.Label(header, text="RL Arcade", style="Title.TLabel").pack(anchor="w")
        self.subtitle = ttk.Label(header, style="Sub.TLabel"); self.subtitle.pack(anchor="w")
        controls = ttk.Frame(self, style="Panel.TFrame", padding=12); controls.pack(fill="x", padx=24, pady=(0,12))
        self.selectors = []
        for label, variable, values, width in (("Environment",self.environment_name,list(ENVIRONMENTS),15), ("Mode",self.mode,("Comparison","Live training"),14)):
            ttk.Label(controls,text=label).pack(side="left",padx=(0,6))
            box = ttk.Combobox(controls,textvariable=variable,values=values,width=width,state="readonly")
            box.pack(side="left",padx=(0,16)); box.bind("<<ComboboxSelected>>",lambda _e:self.load_mode()); self.selectors.append(box)
        self.start_button = ttk.Button(controls,text="Start training",command=self.start_training)
        self.start_button.pack(side="left",padx=4)
        self.stop_button = ttk.Button(controls,text="Stop",command=self.stop_training); self.stop_button.pack(side="left",padx=4)
        self.pause_button = ttk.Button(controls,text="Pause viewer",command=self.toggle_pause); self.pause_button.pack(side="left",padx=4)
        ttk.Button(controls,text="Watch MuJoCo",command=self.watch_biped).pack(side="right",padx=4)
        content = ttk.Frame(self); content.pack(fill="both",expand=True,padx=24,pady=(0,24))
        self.stage = ttk.Frame(content); self.stage.pack(side="left",fill="both",expand=True)
        self.canvases = []
        for title in ("TRAINED DQN", "RANDOM BASELINE"):
            panel = ttk.Frame(self.stage,style="Panel.TFrame"); panel.pack(side="left",fill="both",expand=True,padx=(0,6))
            ttk.Label(panel,text=title,foreground=COLORS["muted"]).pack(pady=(8,0))
            canvas = tk.Canvas(panel,bg=COLORS["panel"],highlightthickness=0); canvas.pack(fill="both",expand=True)
            self.canvases.append((panel,canvas))
        sidebar = ttk.Frame(content,style="Panel.TFrame",width=245,padding=18); sidebar.pack(side="right",fill="y",padx=(6,0)); sidebar.pack_propagate(False)
        self.metrics = {name:tk.StringVar(value="—") for name in ("Trained score","Random score","Episode","Epsilon")}
        for name,var in self.metrics.items():
            ttk.Label(sidebar,text=name.upper(),foreground=COLORS["muted"]).pack(anchor="w",pady=(9,0))
            ttk.Label(sidebar,textvariable=var,style="Metric.TLabel").pack(anchor="w")
        ttk.Separator(sidebar).pack(fill="x",pady=18)
        ttk.Label(sidebar,text="RECENT EPISODES",foreground=COLORS["muted"]).pack(anchor="w")
        self.chart = tk.Canvas(sidebar,width=205,height=120,bg=COLORS["panel"],highlightthickness=0); self.chart.pack(pady=8)
        self.status = tk.StringVar(value="Ready"); ttk.Label(sidebar,textvariable=self.status,foreground=COLORS["muted"],wraplength=205).pack(anchor="w",side="bottom")

    def load_mode(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Training is running", "Stop training before changing modes or environments.")
            return
        training = self.mode.get() == "Live training"
        self.canvases[1][0].pack_forget() if training else self.canvases[1][0].pack(side="left",fill="both",expand=True,padx=(6,0))
        self.start_button.configure(state="normal" if training else "disabled")
        self.stop_button.configure(state="disabled")
        self.subtitle.configure(text="Watch freshly trained episodes while learning continues." if training else "Compare the trained policy with a random baseline side by side.")
        self.status.set("Press Start training; no training runs automatically." if training else "Both policies use isolated, repeatable environment random streams.")
        self.playback.clear(); self._load_comparison() if not training else self._clear_training()

    def _load_comparison(self):
        name = self.environment_name.get(); cls = ENVIRONMENTS[name]; state_size = len(cls(seed=7).reset())
        model = resource_path("models", name.lower().replace(" ","_")+".pt")
        self.runners = [Runner(cls(),DQNPolicy.load(model,state_size,len(cls.action_names)),7), Runner(cls(),RandomAgent(len(cls.action_names),11),7)]

    def _clear_training(self):
        self.runners = []; self.training_stats = {"episode":0,"epsilon":.9999,"score":0,"reward":0.}; self.training_scores = []
        for _panel,canvas in self.canvases: canvas.delete("all")
        self.render_metrics()

    def start_training(self):
        if self.mode.get() != "Live training" or (self.worker and self.worker.is_alive()): return
        self.stop_event.clear(); self.playback_queue.clear(); self.playback.clear(); self.renderer_ready["ready"] = True
        name = self.environment_name.get(); self.worker = threading.Thread(target=self._train_worker,args=(name,),daemon=True); self.worker.start()
        self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal")
        for selector in self.selectors: selector.configure(state="disabled")
        self.status.set("Training from scratch in the background; replaying sampled episodes.")

    def _train_worker(self, name):
        env = ENVIRONMENTS[name](); state_size = len(env.reset()); action_size = len(env.action_names)
        agent = Agent(state_size,action_size); buffer = Buffer(alpha=.6,beta=.4,beta_increment=.001,epsilon=1e-5)
        decay, recorder_class = TRAINING[name]; recorder = recorder_class(self.playback_queue,self.renderer_ready,self.telemetry_lock)
        epsilon, sync_steps = .9999, 0
        for episode in range(1,5001):
            if self.stop_event.is_set(): break
            sync_steps,_unused,reward,score = train_loop(env,agent,buffer,epsilon,action_size,sync_steps,recorder)
            epsilon = max(.01,epsilon*decay); self.progress.put((episode,epsilon,score,reward))
        root = Path(os.getenv("LOCALAPPDATA",Path.cwd()))/"RLArcade"; root.mkdir(parents=True,exist_ok=True)
        torch.save({"training_network":agent.training_network.state_dict(),"frozen_network":agent.frozen_network.state_dict(),"steps":agent.steps},root/(name.lower().replace(" ","_")+"_live.pt"))
        self.progress.put(("done",episode,epsilon,str(root)))

    def stop_training(self):
        self.stop_event.set(); self.status.set("Stopping after the current episode…")

    def tick(self):
        self._read_progress()
        if not self.paused:
            if self.mode.get()=="Comparison":
                for _ in range(self.speed.get()):
                    for runner in self.runners: runner.step()
                for (_panel,canvas),runner in zip(self.canvases,self.runners): self.render_env(canvas,runner.env)
            else: self._play_training_frame()
            self.render_metrics()
        self.after(35,self.tick)

    def _read_progress(self):
        try:
            while True:
                item = self.progress.get_nowait()
                if item[0]=="done":
                    self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled")
                    for selector in self.selectors: selector.configure(state="readonly")
                    self.status.set(f"Training ended at episode {item[1]}. Checkpoint saved in {item[3]}.")
                else:
                    self.training_stats = dict(zip(("episode","epsilon","score","reward"),item)); self.training_scores.append(item[2]); self.training_scores = self.training_scores[-20:]
        except Empty: pass

    def _play_training_frame(self):
        if not self.playback and self.playback_queue:
            self.playback = deque(self.playback_queue.popleft())
        if self.playback:
            self.render_snapshot(self.canvases[0][1],self.playback.popleft())
            if not self.playback:
                with self.telemetry_lock: self.renderer_ready["ready"] = True

    def render_env(self,canvas,env):
        if self.environment_name.get()=="Flappy Bird":
            snap={"bird":(env.x,env.y),"obstacles":env.obstacles,"obstacle_width":env.obstacle_width,"obstacle_gap":env.gap_size,"bird_size":env.bird_size}
        else: snap={"head":env.head,"body":env.body,"apple":env.apple}
        self.render_snapshot(canvas,snap)

    def render_snapshot(self,canvas,snap):
        canvas.delete("all"); width=max(1,canvas.winfo_width()); height=max(1,canvas.winfo_height())
        if "bird" in snap:
            canvas.configure(bg=COLORS["sky"]); sx,sy=width/600,height/600; x,y=snap["bird"]; size=snap["bird_size"]
            for ox,gap,_passed in snap["obstacles"]:
                canvas.create_rectangle(ox*sx,0,(ox+snap["obstacle_width"])*sx,height-(gap+snap["obstacle_gap"])*sy,fill="#2ecb70",outline="")
                canvas.create_rectangle(ox*sx,height-gap*sy,(ox+snap["obstacle_width"])*sx,height,fill="#2ecb70",outline="")
            canvas.create_oval((x-size/2)*sx,height-(y+size/2)*sy,(x+size/2)*sx,height-(y-size/2)*sy,fill="#ffdf45",outline="#7d6720")
        else:
            canvas.configure(bg="#151b25"); size=min(width,height)*.92; cell=size/17; left=(width-size)/2; top=(height-size)/2
            for index in range(18):
                canvas.create_line(left+index*cell,top,left+index*cell,top+size,fill="#222c39")
                canvas.create_line(left,top+index*cell,left+size,top+index*cell,fill="#222c39")
            for x,y in snap["body"]: canvas.create_rectangle(left+x*cell+1,top+(16-y)*cell+1,left+(x+1)*cell-1,top+(17-y)*cell-1,fill=COLORS["accent"] if (x,y)==snap["head"] else "#279b6c",outline="")
            x,y=snap["apple"]; canvas.create_oval(left+x*cell+3,top+(16-y)*cell+3,left+(x+1)*cell-3,top+(17-y)*cell-3,fill=COLORS["danger"],outline="")

    def render_metrics(self):
        if self.mode.get()=="Comparison" and self.runners:
            self.metrics["Trained score"].set(self.runners[0].env.score); self.metrics["Random score"].set(self.runners[1].env.score)
            values=[score for score,_reward in self.runners[0].episodes]
            self.metrics["Episode"].set(len(self.runners[0].episodes)); self.metrics["Epsilon"].set("0 (evaluation)")
        else:
            s=self.training_stats; self.metrics["Trained score"].set(s["score"]); self.metrics["Random score"].set("—")
            self.metrics["Episode"].set(f'{s["episode"]} / 5000'); self.metrics["Epsilon"].set(f'{s["epsilon"]:.4f}'); values=self.training_scores
        self.chart.delete("all")
        if values:
            high=max(1,max(values)); points=[]
            for i,value in enumerate(values): points.extend((5+i*195/max(1,len(values)-1),110-value*100/high))
            if len(points)>2:self.chart.create_line(*points,fill=COLORS["accent"],width=2,smooth=True)
        else:self.chart.create_text(102,60,text="Episode scores appear here",fill=COLORS["muted"])

    def toggle_pause(self):
        self.paused=not self.paused; self.pause_button.configure(text="Resume viewer" if self.paused else "Pause viewer")

    def _biped_command(self,*args):
        if getattr(sys,"frozen",False): candidate=Path(sys.executable).with_name("BipedDemo.exe"); return candidate,[str(candidate),*args]
        candidate=resource_path("biped_demo.py"); return candidate,[sys.executable,str(candidate),*args]

    def watch_biped(self):
        candidate,command=self._biped_command()
        if candidate.exists(): subprocess.Popen(command)
        else: messagebox.showinfo("MuJoCo demo","Place BipedDemo.exe beside RLArcade.exe.")

    def close(self):
        self.stop_event.set(); self.destroy()


if __name__=="__main__": RLArcade().mainloop()
