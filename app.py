import tkinter as tk
import subprocess
import sys
from pathlib import Path
from tkinter import messagebox, ttk

from arcade.agents import DQNPolicy, RandomAgent
from arcade.environments import ENVIRONMENTS
from arcade.resources import resource_path


COLORS = {
    "background": "#11151c",
    "panel": "#1b2430",
    "panel_light": "#263241",
    "text": "#f4f7fb",
    "muted": "#9eabc0",
    "accent": "#52e0a3",
    "danger": "#ff657a",
    "sky": "#7bc8f6",
}


class RLArcade(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL Arcade")
        self.geometry("1060x720")
        self.minsize(900, 620)
        self.configure(bg=COLORS["background"])
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.environment_name = tk.StringVar(value="Flappy Bird")
        self.agent_name = tk.StringVar(value="Trained DQN")
        self.speed = tk.IntVar(value=2)
        self.paused = False
        self.episode_rewards = []
        self.current_reward = 0.0
        self.last_action = "—"
        self.env = None
        self.agent = None

        self._configure_style()
        self._build_ui()
        self.load_environment()
        self.after(35, self.tick)

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=COLORS["background"], foreground=COLORS["text"], font=("Segoe UI Semibold", 24))
        style.configure("Sub.TLabel", background=COLORS["background"], foreground=COLORS["muted"], font=("Segoe UI", 10))
        style.configure("Metric.TLabel", background=COLORS["panel"], foreground=COLORS["accent"], font=("Segoe UI Semibold", 18))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 8), background=COLORS["panel_light"], foreground=COLORS["text"])
        style.map("TButton", background=[("active", "#35465b")])
        style.configure("TCombobox", fieldbackground=COLORS["panel_light"], foreground=COLORS["text"], padding=6)

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=24, pady=(18, 12))
        ttk.Label(header, text="RL Arcade", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Watch a random policy fail, then switch to a Q-learning agent.", style="Sub.TLabel").pack(anchor="w")

        controls = ttk.Frame(self, style="Panel.TFrame", padding=12)
        controls.pack(fill="x", padx=24, pady=(0, 12))
        ttk.Label(controls, text="Environment").pack(side="left", padx=(0, 6))
        environment = ttk.Combobox(controls, textvariable=self.environment_name, values=list(ENVIRONMENTS), width=16, state="readonly")
        environment.pack(side="left", padx=(0, 18))
        environment.bind("<<ComboboxSelected>>", lambda _event: self.load_environment())
        ttk.Label(controls, text="Policy").pack(side="left", padx=(0, 6))
        agent = ttk.Combobox(controls, textvariable=self.agent_name, values=("Trained DQN", "Random agent"), width=17, state="readonly")
        agent.pack(side="left", padx=(0, 18))
        agent.bind("<<ComboboxSelected>>", lambda _event: self.load_agent())
        ttk.Button(controls, text="Restart", command=self.restart).pack(side="left", padx=4)
        self.pause_button = ttk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Launch MuJoCo Biped", command=self.launch_biped).pack(side="right", padx=4)

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        self.canvas = tk.Canvas(content, bg=COLORS["panel"], highlightthickness=0, width=720, height=540)
        self.canvas.pack(side="left", fill="both", expand=True)

        sidebar = ttk.Frame(content, style="Panel.TFrame", width=255, padding=18)
        sidebar.pack(side="right", fill="y", padx=(12, 0))
        sidebar.pack_propagate(False)
        self.metric_vars = {name: tk.StringVar(value="—") for name in ("Score", "Reward", "Steps", "Action")}
        for name, variable in self.metric_vars.items():
            ttk.Label(sidebar, text=name.upper(), foreground=COLORS["muted"]).pack(anchor="w", pady=(8, 0))
            ttk.Label(sidebar, textvariable=variable, style="Metric.TLabel").pack(anchor="w")
        ttk.Separator(sidebar).pack(fill="x", pady=18)
        ttk.Label(sidebar, text="LAST 20 EPISODES", foreground=COLORS["muted"]).pack(anchor="w")
        self.chart = tk.Canvas(sidebar, width=215, height=120, bg=COLORS["panel"], highlightthickness=0)
        self.chart.pack(pady=8)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(sidebar, textvariable=self.status, foreground=COLORS["muted"], wraplength=210).pack(anchor="w", side="bottom")

    def load_environment(self):
        self.env = ENVIRONMENTS[self.environment_name.get()](seed=7)
        self.episode_rewards.clear()
        self.current_reward = 0.0
        self.load_agent()
        self.render()

    def load_agent(self):
        if not self.env:
            return
        if self.agent_name.get() == "Random agent":
            self.agent = RandomAgent(len(self.env.action_names), seed=11)
            self.status.set("Random actions — a useful baseline.")
        else:
            filename = self.environment_name.get().lower().replace(" ", "_") + ".pt"
            model_path = resource_path("models", filename)
            if model_path.exists():
                state_size = len(self.env.reset())
                self.agent = DQNPolicy.load(model_path, state_size, len(self.env.action_names))
                self.status.set("Loaded the trained Dueling Double DQN policy.")
            else:
                self.agent = RandomAgent(len(self.env.action_names), seed=11)
                self.status.set("The trained DQN checkpoint is missing.")
        self.restart()

    def restart(self):
        if not self.env:
            return
        self.state = self.env.reset()
        self.current_reward = 0.0
        self.last_action = "—"
        self.render()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")

    def tick(self):
        if not self.paused and self.env and self.agent:
            for _ in range(self.speed.get()):
                action = self.agent.select_action(self.state)
                self.last_action = self.env.action_names[action]
                self.state, reward, done = self.env.step(action)
                self.current_reward += reward
                if done:
                    self.episode_rewards.append(self.current_reward)
                    self.episode_rewards = self.episode_rewards[-20:]
                    self.state = self.env.reset()
                    self.current_reward = 0.0
            self.render()
        self.after(35, self.tick)

    def render(self):
        if not self.env:
            return
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if self.environment_name.get() == "Flappy Bird":
            self._render_flappy(width, height)
        else:
            self._render_snake(width, height)
        self.metric_vars["Score"].set(str(self.env.score))
        self.metric_vars["Reward"].set(f"{self.current_reward:.2f}")
        self.metric_vars["Steps"].set(f"{self.env.steps:,}")
        self.metric_vars["Action"].set(self.last_action)
        self._render_chart()

    def _render_flappy(self, width, height):
        self.canvas.configure(bg=COLORS["sky"])
        sx, sy = width / self.env.width, height / self.env.height
        self.canvas.create_rectangle(0, height - 8, width, height, fill="#d9b96e", outline="")
        for x, gap, _passed in self.env.obstacles:
            left, right = x * sx, (x + 50) * sx
            gap_top = height - (gap + 150) * sy
            gap_bottom = height - gap * sy
            self.canvas.create_rectangle(left, 0, right, gap_top, fill="#2ecb70", outline="#197d46", width=2)
            self.canvas.create_rectangle(left, gap_bottom, right, height, fill="#2ecb70", outline="#197d46", width=2)
        x, y = self.env.x * sx, height - self.env.y * sy
        self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill="#ffdf45", outline="#7d6720", width=2)
        self.canvas.create_oval(x + 4, y - 6, x + 8, y - 2, fill="#11151c", outline="")

    def _render_snake(self, width, height):
        self.canvas.configure(bg="#151b25")
        size = min(width, height) * 0.92
        cell = size / 17
        ox, oy = (width - size) / 2, (height - size) / 2
        for index in range(18):
            self.canvas.create_line(ox + index * cell, oy, ox + index * cell, oy + size, fill="#222c39")
            self.canvas.create_line(ox, oy + index * cell, ox + size, oy + index * cell, fill="#222c39")
        for index, (x, y) in enumerate(self.env.body):
            color = COLORS["accent"] if index == 0 else "#279b6c"
            self.canvas.create_rectangle(ox + x * cell + 1, oy + (16 - y) * cell + 1, ox + (x + 1) * cell - 1, oy + (17 - y) * cell - 1, fill=color, outline="")
        x, y = self.env.apple
        self.canvas.create_oval(ox + x * cell + 3, oy + (16 - y) * cell + 3, ox + (x + 1) * cell - 3, oy + (17 - y) * cell - 3, fill=COLORS["danger"], outline="")

    def _render_chart(self):
        self.chart.delete("all")
        values = self.episode_rewards
        if not values:
            self.chart.create_text(107, 60, text="Complete an episode\nto see rewards", fill=COLORS["muted"], justify="center")
            return
        low, high = min(values), max(values)
        span = max(1.0, high - low)
        points = []
        for index, value in enumerate(values):
            x = 5 + index * 205 / max(1, len(values) - 1)
            y = 110 - (value - low) * 100 / span
            points.extend((x, y))
        if len(points) >= 4:
            self.chart.create_line(*points, fill=COLORS["accent"], width=2, smooth=True)
        else:
            self.chart.create_oval(points[0] - 2, points[1] - 2, points[0] + 2, points[1] + 2, fill=COLORS["accent"], outline="")

    def launch_biped(self):
        if getattr(sys, "frozen", False):
            candidate = Path(sys.executable).with_name("BipedDemo.exe")
            command = [str(candidate)]
        else:
            candidate = resource_path("biped_demo.py")
            command = [sys.executable, str(candidate)]
        if not candidate.exists():
            messagebox.showinfo(
                "Optional MuJoCo demo",
                "Place BipedDemo.exe beside RLArcade.exe to enable the live biped demo. "
                "It is distributed separately because the MuJoCo and PyTorch runtime is much larger.",
            )
            return
        try:
            subprocess.Popen(command)
            self.status.set("Launched the optional MuJoCo biped demo.")
        except OSError as error:
            messagebox.showerror("Could not launch biped", str(error))

if __name__ == "__main__":
    RLArcade().mainloop()
