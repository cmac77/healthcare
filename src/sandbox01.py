# %%
import tkinter as tk

# from tkinter import StringVar
# import sys
# from pathlib import Path
# import tkinter as tk

root = tk.Tk()
root.title(f"Recode and Order Column")

# Get screen dimensions for dynamic sizing
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Fixed window dimensions
window_height = int(screen_height * 0.5)  # 30% of the screen height
window_width = int(
    screen_height * 0.5
)  # 20% of the screen height (as requested)

# Apply fixed window size
root.geometry(f"{window_width}x{window_height}+50+50")

# Force window to the front
root.lift()
root.attributes("-topmost", True)
root.mainloop()

# # %%
# # Layout for Yes/No and Confirm buttons
# top_frame = tk.Frame(root)
# top_frame.pack(pady=10)

# # Label with text wrapping
# label_text = f"Do you want to recode '{column_name}' as an ordinal variable?"
# label = tk.Label(
#     top_frame,
#     text=label_text,
#     wraplength=window_width - 50,
#     justify="left",
# )
# label.grid(row=0, column=0, columnspan=2)

# # Stack Yes/No buttons
# recode_var = StringVar(value="no")
# yes_button = tk.Radiobutton(
#     top_frame, text="Yes", variable=recode_var, value="yes"
# )
# no_button = tk.Radiobutton(
#     top_frame, text="No", variable=recode_var, value="no"
# )
# yes_button.grid(row=1, column=0, pady=5)
# no_button.grid(row=2, column=0, pady=5)

# # Confirm button
# confirm_button = tk.Button(
#     top_frame, text="Confirm", command=lambda: root.quit()
# )
# confirm_button.grid(row=1, column=1, rowspan=2, padx=20, pady=10)

# # Label for listbox
# tk.Label(
#     root, text="Reorder the unique values (top is smallest ordinal):"
# ).pack(pady=10)

# # Create listbox for unique values with a fixed height of 10 items and scrollbar
# listbox_frame = tk.Frame(root)
# listbox_frame.pack(fill="both", expand=True)

# scrollbar = tk.Scrollbar(listbox_frame)
# scrollbar.pack(side="right", fill="y")

# listbox = tk.Listbox(
#     listbox_frame,
#     selectmode=tk.SINGLE,
#     height=10,
#     yscrollcommand=scrollbar.set,
# )  # Fixed height of 10 items
# for val in unique_values:
#     listbox.insert(tk.END, str(val))
# listbox.pack(side="left", fill="both", expand=True)

# scrollbar.config(command=listbox.yview)

# # Bottom buttons (Move Up / Move Down)
# button_frame = tk.Frame(root)
# button_frame.pack(pady=10)
# tk.Button(
#     button_frame,
#     text="Move Up",
#     command=lambda: reorder_listbox(listbox, "up"),
# ).grid(row=0, column=0, padx=5)
# tk.Button(
#     button_frame,
#     text="Move Down",
#     command=lambda: reorder_listbox(listbox, "down"),
# ).grid(row=0, column=1, padx=5)

# # Start the tkinter event loop
# root.mainloop()

# # Retrieve the order before the window is destroyed
# ordered_values = [listbox.get(i) for i in range(listbox.size())]
# root.destroy()

# # Return user selection and ordering
# return ordered_values, recode_var.get()


# def create_window():
#     root = tk.Tk()
#     root.title("Test Window")

#     # Get screen dimensions
#     screen_width = root.winfo_screenwidth()
#     screen_height = root.winfo_screenheight()

#     # Set window dimensions to 50% of the screen size
#     window_width = int(screen_width * 0.5)
#     window_height = int(screen_height * 0.5)

#     # Set window size using geometry
#     root.geometry(f"{window_width}x{window_height}+0+0")

#     # Lift the window and start the event loop
#     root.lift()
#     root.mainloop()


# create_window()
