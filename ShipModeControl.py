import tkinter as tk
from tkinter import ttk
import math

class VesselDynamicsModel:
    def __init__(self):
        self.length = 90.0
        self.beam = 16
        self.draft = 5
        self.metacentric_height = 0.9
        self.max_speed = 20
        self.recalculate_periods()

    def recalculate_periods(self):
        self.roll_period = (0.8 * self.beam) / math.sqrt(self.metacentric_height)
        self.pitch_period = 2.5 * math.sqrt(self.draft)

    def calculate_apparent_wave_period(self, wave_length, speed_knots, heading_deg):
        v_ms = speed_knots * 0.514
        heading_rad = math.radians(heading_deg)
        wave_speed = 1.25 * math.sqrt(wave_length)
        denom = wave_speed - v_ms * math.cos(heading_rad)
        if abs(denom) < 0.01:
            return 999.0
        return abs(wave_length / denom)

    def calculate_resonance_zones(self, wave_length):
        wave_speed = 1.25 * math.sqrt(wave_length)
        zones = []
        resonances = [
            (self.roll_period, (0.8, 1.2), "orange", 0), 
            (self.roll_period, (1.85, 2.15), "purple", 1), 
            (self.pitch_period, (0.8, 1.2), "blue", 2) 
        ]
        for period, (r_min, r_max), color, offset in resonances:
            t_min = period / r_max
            t_max = period / r_min
            v1 = wave_speed - wave_length / t_min
            v2 = wave_speed - wave_length / t_max
            zones.append({
                "min": min(v1, v2)/0.5144,
                "max": max(v1, v2)/0.5144,
                "color": color,
                "offset": offset
            })
        return zones


class FuzzyDecisionSystem:
    @staticmethod
    def r_function(x, a, b):
        if x <= a: return 0.0
        if x >= b: return 1.0
        return (x - a)/(b - a)

    @staticmethod
    def trimf(x, a, b, c):
        return max(min((x-a)/(b-a+1e-6),(c-x)/(c-b+1e-6)),0)

    def evaluate(self, roll, pitch, r_roll, r_pitch):
        mu_hr = self.r_function(roll, 12, 20)
        mu_hp = self.r_function(pitch, 2.5, 4.5)
        mu_rr_main = self.trimf(r_roll,0.8,1.0,1.2)
        mu_rr_param = self.trimf(r_roll,1.8,1.9,2.1)
        mu_rp_main = self.trimf(r_pitch,0.8,1.0,1.2)
        rule1 = min(mu_hr, mu_rr_main)
        rule2 = min(mu_hr, mu_rr_param)
        rule3 = min(mu_hp, mu_rp_main)
        agg = max(rule1, rule2, rule3)
        num = den = 0.0
        for y in range(0,101,2):
            mu_out = min(FuzzyDecisionSystem.r_function(y,50,100), agg)
            num += y*mu_out
            den += mu_out
        danger = 0 if den==0 else num/den
        return {"danger": danger,"rules":(rule1,rule2,rule3),"level":agg}


class DecisionSupportApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("600x750")
        self.model = VesselDynamicsModel()
        self.fuzzy = FuzzyDecisionSystem()

        self.wave_length = 1.2*self.model.length  
        self.speed = 16        

        self.heading_var = tk.DoubleVar(value=90.0)
        self.roll_var = tk.DoubleVar(value=12.0)
        self.pitch_var = tk.DoubleVar(value=2.5)

        self.setup_style()
        self.create_ui()
        self.run_analysis()

    def setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TLabelFrame.Label", font=("Segoe UI", 11, "bold"))
        style.configure("TEntry", font=("Segoe UI", 10))

    def detect_resonance(self, r_roll, r_pitch):
        resonances = []
        if 0.8 <= r_roll <= 1.2:
            resonances.append("БОРТОВОЙ ОСНОВНОЙ РЕЗОНАНС")
        if 1.85 <= r_roll <= 2.15:
            resonances.append("БОРТОВОЙ ПАРАМЕТРИЧЕСКИЙ РЕЗОНАНС")
        if 0.8 <= r_pitch <= 1.2:
            resonances.append("КИЛЕВОЙ ОСНОВНОЙ РЕЗОНАНС")
        return resonances

    def create_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        grp_wave = ttk.LabelFrame(main, text="Изменяемые параметры", padding=10)
        grp_wave.pack(fill=tk.X, pady=5)

        inputs_frame = ttk.Frame(grp_wave)
        inputs_frame.pack(fill=tk.X)

        lbl_roll = ttk.Label(inputs_frame, text="Амплитуда \nбортовой качки (°):")
        lbl_roll.grid(row=0, column=0, padx=(0, 10), sticky="w")

        lbl_pitch = ttk.Label(inputs_frame, text="Амплитуда \nкилевой качки (°):")
        lbl_pitch.grid(row=0, column=1, padx=(10, 0), sticky="w")

        lbl_heading = ttk.Label(inputs_frame, text="Курсовой \nугол (°):")
        lbl_heading.grid(row=0, column=2, padx=(10, 0), sticky="n")

        def bind_entry(var, min_val, max_val):
            def on_change(e):
                self.validate_and_update(var, min_val, max_val)
            entry = ttk.Entry(inputs_frame, textvariable=var, width=12, font=("Segoe UI", 10))
            entry.bind("<FocusOut>", on_change)
            entry.bind("<Return>", on_change)
            return entry

        entry_roll = bind_entry(self.roll_var, 0, 30)
        entry_roll.grid(row=1, column=0, padx=(0, 10), pady=(5, 0))

        entry_pitch = bind_entry(self.pitch_var, 0, 10)
        entry_pitch.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))

        entry_heading = bind_entry(self.heading_var, 0, 180)
        entry_heading.grid(row=1, column=2, padx=(10, 0), pady=(5, 0))

        grp_r = ttk.LabelFrame(main, text="Результаты анализа", padding=10)
        grp_r.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        results_frame = ttk.Frame(grp_r)
        results_frame.pack(fill=tk.BOTH, expand=True)

        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)

        left_col = ttk.Frame(results_frame)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        ttk.Label(left_col, text="Анализ состояния", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.log_analysis = tk.Text(left_col, height=12, font=("Consolas", 10), bg="#f8f8f8")
        self.log_analysis.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        right_col = ttk.Frame(results_frame)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        ttk.Label(right_col, text="Решение системы", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.log_decision = tk.Text(right_col, height=12, font=("Consolas", 10), bg="#ffffff")
        self.log_decision.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self.log_decision.tag_configure("safe", foreground="#2e7d32", font=("Consolas", 10, "bold"))
        self.log_decision.tag_configure("danger", foreground="#c62828", font=("Consolas", 10, "bold"))

        results_frame.rowconfigure(0, weight=1)
        
        ttk.Label(main, text="Диаграмма резонансных зон", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.canvas = tk.Canvas(main, bg="white", height=400)  
        self.canvas.pack(fill=tk.BOTH, expand=False, pady=(0, 0))  
        self.canvas.bind("<Configure>", lambda e: self.draw_diagram())

    def create_entry_row(self, parent, label_text, var, min_val, max_val):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)

        lbl = ttk.Label(frame, text=f"{label_text} ({min_val}-{max_val}):")
        lbl.pack(anchor=tk.W)

        def on_focus_out(event):
            self.validate_and_update(var, min_val, max_val)

        def on_return(event):
            self.validate_and_update(var, min_val, max_val)

        entry = ttk.Entry(frame, textvariable=var, width=10, font=("Segoe UI", 10))
        entry.bind("<FocusOut>", on_focus_out)
        entry.bind("<Return>", on_return)
        entry.pack(side=tk.LEFT)

    def validate_and_update(self, var, min_val, max_val):
        try:
            value = float(var.get())
            if value < min_val:
                value = min_val
            elif value > max_val:
                value = max_val
            var.set(round(value, 1))
        except (ValueError, TypeError):
            var.set(min_val)
        self.run_analysis()

    def suggest_safe_heading(self, danger):
        if danger < 40:
            return "Курс безопасен"
        base = self.heading_var.get()
        for d in [10, -10, 20, -20, 30, -30, 45, -45, 90, -90]:
            test = base + d
            if test < 0:
                test = abs(test)
            if test > 180:
                test = 180 - (test - 180)
            tau = self.model.calculate_apparent_wave_period(
                self.wave_length,
                self.speed,
                test
            )
            r_roll = self.model.roll_period / tau if tau < 900 else 0
            r_pitch = self.model.pitch_period / tau if tau < 900 else 0
            res = self.fuzzy.evaluate(
                self.roll_var.get(),
                self.pitch_var.get(),
                r_roll, r_pitch
            )
            if res["danger"] < 25:
                return f" \n Рекомендована смена курса на {test:.0f}°"
        return "\nРекомендовано снижение скорости"

    def run_analysis(self):
        wl = self.wave_length
        sp = self.speed
        hd = self.heading_var.get()
        roll = self.roll_var.get()
        pitch = self.pitch_var.get()

        tau = self.model.calculate_apparent_wave_period(wl, sp, hd)
        r_roll = self.model.roll_period / tau if tau < 900 else 0
        r_pitch = self.model.pitch_period / tau if tau < 900 else 0
        fuzzy = self.fuzzy.evaluate(roll, pitch, r_roll, r_pitch)
        danger = fuzzy["danger"]
        resonance_types = self.detect_resonance(r_roll, r_pitch)

        P_E = 0.75
        P_H = 0.9 * P_E + 0.01 * (1 - P_E)
        CF = 0.9

        

        rec = self.suggest_safe_heading(danger)

        log = (f"ТЕКУЩИЕ ПАРАМЕТРЫ:\n"
               f"τ_волны: {tau:.2f} c\n"
               f"Соотн. бортовой: {r_roll:.2f}\n"
               f"Соотн. килевой: {r_pitch:.2f}\n"
               f"ВЕРОЯТНОСТНЫЕ ОЦЕНКИ:\n"
               f"Байеc: {P_H:.4f}\n"
               f"Шортлифф: {CF:.2f}\n"
               f"СТЕПЕНЬ ДОВЕРИЯ ПРАВИЛАМ:\n"
               f"Правило 1: {fuzzy['rules'][0]:.2f}\n"
               f"Правило 2: {fuzzy['rules'][1]:.2f}\n"
               f"Правило 3: {fuzzy['rules'][2]:.2f}\n"
               f"ВЕРОЯТНОСТЬ ОПАСНОСТИ: {danger:.1f}%")

        self.log_analysis.delete(1.0, tk.END)
        self.log_decision.delete(1.0, tk.END)
        self.log_analysis.insert(tk.END, log)

        if danger > 40:
            self.log_decision.insert(tk.END, "ОПАСНАЯ СИТУАЦИЯ", "danger")
            for r in resonance_types:
                self.log_decision.insert(tk.END, f"\n• {r} ", "danger")
            self.log_decision.insert(tk.END, rec, "danger")
        else:
            self.log_decision.insert(tk.END, "Cитуация нормальная.\nУгрозы нет.\nРекомендаций нет", "safe")

        self.draw_diagram()

    def draw_diagram(self):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50:
            return
        cx, cy = w / 2, h - 30
        R = min(cx, cy) * 0.95
        scale = R / self.model.max_speed

        for v in range(5, 25, 5):
            r = v * scale
            self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180, outline="lightgray", style=tk.ARC)
            self.canvas.create_text(cx + r + 5, cy - 5, text=str(v), fill="gray")

        for a in range(0, 181, 30):
            rad = math.radians(a)
            x = cx - R * math.cos(rad)
            y = cy - R * math.sin(rad)
            self.canvas.create_line(cx, cy, x, y, fill="lightgray")
            self.canvas.create_text(x, y - 10, text=f"{a}°")

        zones = self.model.calculate_resonance_zones(self.wave_length)

        step = 2
        for y in range(int(cy - R), int(cy), step):
            dy = cy - y
            if R**2 - dy**2 < 0:
                continue
            hw = math.sqrt(R**2 - dy**2)
            left_limit, right_limit = cx - hw, cx + hw
            for z in zones:
                x1 = cx - z["max"] * scale
                x2 = cx - z["min"] * scale
                lx = max(min(x1, x2), left_limit)
                rx = min(max(x1, x2), right_limit)
                if lx < rx:
                    y_offset = y + z["offset"] * step
                    self.canvas.create_rectangle(
                        lx, y_offset, rx, y_offset + step,
                        fill=z["color"],
                        outline="",
                        stipple="gray50"
                    )

        sp = self.speed
        hd = self.heading_var.get()
        rad = math.radians(hd)
        sx = cx - sp * scale * math.cos(rad)
        sy = cy - sp * scale * math.sin(rad)
        self.canvas.create_line(cx, cy, sx, sy, arrow=tk.LAST, width=2)
        self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="black", outline="white", width=2)

        colors = [("orange", "Осн. бор."), ("purple", "Пар. бор."), ("blue", "Осн. кил.")]
        for i, (col, text) in enumerate(colors):
            self.canvas.create_rectangle(20, 20 + i * 25, 35, 35 + i * 25, fill=col, outline="", stipple="gray25")
            self.canvas.create_text(40, 27 + i * 25, text=text, anchor=tk.W)


if __name__ == "__main__":
    DecisionSupportApp().mainloop()
