import tkinter as tk


def create_window():
    root = tk.Tk()
    root.title("Test Window")

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window dimensions to 50% of the screen size
    window_width = int(screen_width * 0.5)
    window_height = int(screen_height * 0.5)

    # Set window size using geometry
    root.geometry(f"{window_width}x{window_height}+50+50")

    # Lift the window and start the event loop
    root.lift()
    root.attributes("-topmost", True)
    root.mainloop()


create_window()
