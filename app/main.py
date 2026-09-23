import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys


class FaceRecognitionApp:

    def __init__(self, root):
        self.root = root

        # Window settings
        self.root.title("AI Face Recognition System")
        self.root.geometry("900x600")
        self.root.configure(bg="#101114")
        self.root.resizable(False, False)

        # =========================
        # TITLE
        # =========================

        title = tk.Label(
            self.root,
            text="AI FACE RECOGNITION",
            font=("Helvetica", 32, "bold"),
            fg="white",
            bg="#101114"
        )

        title.pack(pady=(80, 10))

        # =========================
        # SUBTITLE
        # =========================

        subtitle = tk.Label(
            self.root,
            text="AI-powered personalized greeting system",
            font=("Helvetica", 16),
            fg="#aaaaaa",
            bg="#101114"
        )

        subtitle.pack(pady=(0, 50))

        # =========================
        # START RECOGNITION BUTTON
        # =========================

        start_button = tk.Button(
            self.root,
            text="START RECOGNITION",
            font=("Helvetica", 18, "bold"),
            width=25,
            height=2,
            bg="#2d7ff9",
            fg="white",
            activebackground="#1f66d0",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.start_recognition
        )

        start_button.pack(pady=12)

        # =========================
        # MANAGE PEOPLE BUTTON
        # =========================

        people_button = tk.Button(
            self.root,
            text="MANAGE PEOPLE",
            font=("Helvetica", 16, "bold"),
            width=25,
            height=2,
            bg="#33363d",
            fg="white",
            activebackground="#454952",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.manage_people
        )

        people_button.pack(pady=12)

        # =========================
        # EXIT BUTTON
        # =========================

        exit_button = tk.Button(
            self.root,
            text="EXIT",
            font=("Helvetica", 14),
            width=15,
            height=1,
            bg="#24262b",
            fg="white",
            activebackground="#3a3d43",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.root.destroy
        )

        exit_button.pack(pady=25)

        # =========================
        # STATUS
        # =========================

        self.status = tk.Label(
            self.root,
            text="● System Ready",
            font=("Helvetica", 13),
            fg="#55cc88",
            bg="#101114"
        )

        self.status.pack(
            side="bottom",
            pady=20
        )

    # ==================================================
    # START FACE RECOGNITION
    # ==================================================

    def start_recognition(self):

        self.status.config(
            text="● Starting AI recognition...",
            fg="#ffaa00"
        )

        self.root.update()

        try:

            # Get project folder
            project_folder = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )

            # Locate live_recognition.py
            script_path = os.path.join(
                project_folder,
                "live_recognition.py"
            )

            # Check that the file exists
            if not os.path.exists(script_path):

                messagebox.showerror(
                    "File Not Found",
                    "Could not find live_recognition.py\n\n"
                    + script_path
                )

                self.status.config(
                    text="● Recognition unavailable",
                    fg="#ff5555"
                )

                return

            # Start recognition using the current Python environment
            subprocess.Popen(
                [
                    sys.executable,
                    script_path
                ],
                cwd=project_folder
            )

            self.status.config(
                text="● Recognition Running",
                fg="#55cc88"
            )

        except Exception as error:

            messagebox.showerror(
                "Recognition Error",
                f"Could not start recognition.\n\n{error}"
            )

            self.status.config(
                text="● System Error",
                fg="#ff5555"
            )

    # ==================================================
    # MANAGE PEOPLE
    # ==================================================

    def manage_people(self):

        messagebox.showinfo(
            "Manage People",
            "People management will be added next."
        )


# ======================================================
# MAIN
# ======================================================

def main():

    root = tk.Tk()

    app = FaceRecognitionApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()