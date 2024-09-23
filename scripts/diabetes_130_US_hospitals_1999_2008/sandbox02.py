import tkinter as tk
import pandas as pd
import numpy as np


class ColumnEditor:
    def __init__(
        self,
        df,
        window_width_ratio=0.25,
        window_height_ratio=0.25,
        top_panel_height_ratio=0.1,
        bottom_panel_height_ratio=0.1,
        root_bg_color="orange",  # Add root window background color
        top_panel_color="lightgrey",
        middle_panel_color="lightblue",
        bottom_panel_color="lightgreen",
        listbox_top_color="white",
        listbox_bottom_color="lightpink",
        button_frame_color="orange",
        button_bg_color="yellow",
        button_fg_color="black",
        divider_color="purple",
        scrollbar_color="brown",
    ):
        self.df = df
        self.current_column_index = 0
        self.root = tk.Tk()
        self.root.title("Column Editor")
        self.is_editing = tk.BooleanVar(value=False)

        # Set default or user-specified panel sizes and colors
        self.window_width_ratio = window_width_ratio
        self.window_height_ratio = window_height_ratio
        self.top_panel_height_ratio = top_panel_height_ratio
        self.bottom_panel_height_ratio = bottom_panel_height_ratio
        self.root_bg_color = (
            root_bg_color  # Add root window background color initialization
        )
        self.top_panel_color = top_panel_color
        self.middle_panel_color = middle_panel_color
        self.bottom_panel_color = bottom_panel_color
        self.listbox_top_color = listbox_top_color
        self.listbox_bottom_color = listbox_bottom_color
        self.button_frame_color = button_frame_color
        self.button_bg_color = button_bg_color
        self.button_fg_color = button_fg_color
        self.divider_color = divider_color
        self.scrollbar_color = scrollbar_color

        self.create_ui_window()

    def create_ui_window(self):

        # Set the background color of the Tkinter root window using the initialized value
        self.root.configure(bg=self.root_bg_color)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set window dimensions based on ratios
        window_width = int(screen_width * self.window_width_ratio)
        total_height = int(screen_height * self.window_height_ratio)

        self.root.geometry(f"{window_width}x{total_height}+50+50")

        # Divide the total height between the top, middle, and bottom panels
        top_panel_height = int(total_height * self.top_panel_height_ratio)
        bottom_panel_height = int(
            total_height * self.bottom_panel_height_ratio
        )
        middle_panel_height = total_height - (
            top_panel_height + bottom_panel_height
        )

        # Top Panel (Next Column Button, Confirm Text Edit Button, and Recode Button)
        top_panel = tk.Frame(
            self.root, height=top_panel_height, bg=self.top_panel_color
        )
        top_panel.pack(fill="x", expand=False)

        button_frame = tk.Frame(
            top_panel, bg=self.button_frame_color
        )  # Frame color for buttons
        button_frame.place(
            relx=0.5, rely=0.5, anchor="center"
        )  # Explicitly centering the frame

        self.next_button = tk.Button(
            button_frame,
            text="Next Column",
            command=self.next_column,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.next_button.pack(side="left", padx=10)

        self.confirm_button = tk.Button(
            button_frame,
            text="Confirm Text Edit",
            command=self.confirm_changes,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.confirm_button.pack(side="left", padx=10)

        self.recode_button = tk.Button(
            button_frame,
            text="Recode as Ordinal",
            command=self.recode_values_as_ordinal,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.recode_button.pack(side="left", padx=10)

        # Middle Panel (with two listboxes)
        middle_panel = tk.Frame(
            self.root, height=middle_panel_height, bg=self.middle_panel_color
        )
        middle_panel.pack(fill="x", expand=True)

        # Divide the middle panel into two listboxes: top for column name, bottom for unique values
        top_listbox_height = int(
            middle_panel_height * 0.2
        )  # 20% for column name
        bottom_listbox_height = (
            middle_panel_height - top_listbox_height
        )  # 80% for unique values

        # Top Listbox (for column name)
        self.top_listbox_frame = tk.Frame(
            middle_panel, height=top_listbox_height, bg=self.middle_panel_color
        )
        self.top_listbox_frame.pack(fill="x")
        self.top_listbox = tk.Listbox(
            self.top_listbox_frame,
            selectmode="extended",
            height=1,
            bg=self.listbox_top_color,
            bd=1,
            relief="solid",  # Background color, border width, border style
            highlightthickness=2,  # Border thickness around the listbox
            highlightbackground="red",  # Border color
            highlightcolor="blue",  # Border color when the listbox is focused
        )
        self.top_listbox.pack(fill="both", expand=True)
        self.top_listbox.bind("<Double-1>", self.edit_top_listbox)

        # Divider between the two listboxes
        divider = tk.Frame(
            middle_panel, height=2, bg=self.divider_color
        )  # Divider color
        divider.pack(fill="x")

        # Bottom Listbox (for unique values)
        self.bottom_listbox_frame = tk.Frame(
            middle_panel,
            height=bottom_listbox_height,
            bg=self.middle_panel_color,
        )
        self.bottom_listbox_frame.pack(fill="x", expand=True)
        self.bottom_listbox = tk.Listbox(
            self.bottom_listbox_frame,
            selectmode="extended",
            height=10,
            bg=self.listbox_bottom_color,
            bd=1,
            relief="solid",  # Background color, border width, border style
            highlightthickness=2,  # Border thickness around the listbox
            highlightbackground="red",  # Border color when not focused
            highlightcolor="blue",  # Border color when focused
        )
        self.bottom_listbox.pack(side="left", fill="both", expand=True)
        self.bottom_listbox.bind("<Double-1>", self.edit_bottom_listbox)

        scrollbar = tk.Scrollbar(
            self.bottom_listbox_frame,
            command=self.bottom_listbox.yview,
            bg=self.scrollbar_color,
        )
        scrollbar.pack(side="right", fill="y")
        self.bottom_listbox.config(yscrollcommand=scrollbar.set)

        # Bottom Panel (Move Up/Move Down Buttons)
        bottom_panel = tk.Frame(
            self.root, height=bottom_panel_height, bg=self.bottom_panel_color
        )
        bottom_panel.pack(fill="x")

        button_frame = tk.Frame(
            bottom_panel, bg=self.button_frame_color
        )  # Frame color for bottom buttons
        button_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.move_up_button = tk.Button(
            button_frame,
            text="Move Up",
            command=self.move_up,
            bg=self.button_bg_color,
            fg=self.button_fg_color,
        )
        self.move_up_button.pack(side="left", padx=10)

        self.move_down_button = tk.Button(
            button_frame,
            text="Move Down",
            command=self.move_down,
            bg=self.button_bg_color,
            fg=self.button_fg_color,
        )
        self.move_down_button.pack(side="left", padx=10)

        # Load the first column
        self.load_column(0)

        # Start the Tkinter event loop
        self.root.mainloop()

    def load_column(self, col_index):
        self.current_column_index = col_index
        col_name = self.df.columns[col_index]

        # Check if the column is numerical
        is_numeric = pd.api.types.is_numeric_dtype(self.df[col_name])

        # Clear the listboxes
        self.top_listbox.delete(0, tk.END)
        self.bottom_listbox.delete(0, tk.END)

        # Insert column name in the top listbox
        self.top_listbox.insert(tk.END, col_name)

        # If the column is non-numerical, insert unique values
        if not is_numeric:
            unique_values = self.df[col_name].unique()
            for value in unique_values:
                self.bottom_listbox.insert(tk.END, value)
        else:
            self.bottom_listbox.insert(
                tk.END, "Numerical column - values not editable"
            )

    def next_column(self):
        next_col_index = (self.current_column_index + 1) % len(self.df.columns)
        self.load_column(next_col_index)

    def move_up(self):
        if self.is_editing.get():
            return

        selected_indices = list(self.bottom_listbox.curselection())
        if not selected_indices or selected_indices[0] == 0:
            return

        for index in selected_indices:
            if index > 0:
                value = self.bottom_listbox.get(index)
                self.bottom_listbox.delete(index)
                self.bottom_listbox.insert(index - 1, value)

        new_selection = [i - 1 for i in selected_indices]
        for index in new_selection:
            self.bottom_listbox.selection_set(index)

    def move_down(self):
        if self.is_editing.get():
            return

        selected_indices = list(self.bottom_listbox.curselection())
        if (
            not selected_indices
            or selected_indices[-1] == self.bottom_listbox.size() - 1
        ):
            return

        for index in reversed(selected_indices):
            if index < self.bottom_listbox.size() - 1:
                value = self.bottom_listbox.get(index)
                self.bottom_listbox.delete(index)
                self.bottom_listbox.insert(index + 1, value)

        new_selection = [i + 1 for i in selected_indices]
        for index in new_selection:
            self.bottom_listbox.selection_set(index)

    def edit_top_listbox(self, event):
        index = 0  # Only one item in the top listbox
        self.edit_listbox_item(self.top_listbox, index, is_top=True)

    def edit_bottom_listbox(self, event):
        try:
            index = self.bottom_listbox.curselection()[0]
            if "Numerical column" in self.bottom_listbox.get(index):
                return  # Prevent editing if the column is numerical
            self.edit_listbox_item(self.bottom_listbox, index)
        except IndexError:
            return  # No item selected

    def edit_listbox_item(self, listbox, index, is_top=False):
        if self.is_editing.get():
            return

        # Set editing mode
        self.is_editing.set(True)

        # Get bounding box of the selected item
        bbox = listbox.bbox(index)
        if bbox:
            x, y, width, height = bbox

            value = listbox.get(index)
            entry = tk.Entry(listbox, width=width)
            entry.insert(0, value)
            entry.place(x=x, y=y)

            def save_edit(event=None):
                new_value = entry.get().strip()

                if is_top:
                    old_col_name = self.df.columns[self.current_column_index]
                    self.df.rename(
                        columns={old_col_name: new_value}, inplace=True
                    )
                else:
                    if new_value == "":
                        new_value = np.nan  # Convert empty input to NaN
                    old_value = listbox.get(index)

                    # **Direct assignment in DataFrame** to propagate changes correctly
                    col_name = self.df.columns[self.current_column_index]
                    mask = self.df[col_name] == old_value
                    self.df.loc[mask, col_name] = new_value

                listbox.delete(index)
                listbox.insert(index, new_value)
                entry.destroy()
                self.is_editing.set(False)

            def cancel_edit(event=None):
                entry.destroy()
                self.is_editing.set(False)

            entry.bind("<Return>", save_edit)
            entry.bind("<Escape>", cancel_edit)
            entry.bind("<FocusOut>", save_edit)
            entry.focus_set()

    def recode_values_as_ordinal(self):
        col_name = self.df.columns[self.current_column_index]
        is_numeric = pd.api.types.is_numeric_dtype(self.df[col_name])

        if not is_numeric:
            # Get the current order of unique values in the bottom listbox, skipping NaNs
            seen_values = (
                set()
            )  # To keep track of values that have been assigned ordinals
            unique_values = []
            for i in range(self.bottom_listbox.size()):
                value = self.bottom_listbox.get(i)
                if value not in seen_values and pd.notna(
                    value
                ):  # Skip NaN values
                    unique_values.append(value)
                    seen_values.add(value)

            # Create a mapping of value to ordinal based on the listbox order
            recode_map = {value: i for i, value in enumerate(unique_values)}
            self.df[col_name] = (
                self.df[col_name].map(recode_map).fillna(self.df[col_name])
            )

            print(
                f"Column '{col_name}' recoded as ordinal with mapping {recode_map}"
            )
        else:
            print("Numerical columns cannot be recoded as ordinal.")

        # Reload the column after recoding (for numerical columns)
        self.load_column(self.current_column_index)

    def confirm_changes(self):
        print("Updated DataFrame:")
        print(self.df)


# Function to create a DataFrame with mixed types of columns
def create_random_df(num_rows, num_cols):
    column_names = [f"Column {i+1}" for i in range(num_cols)]
    data = {}
    for i in range(num_cols):
        if i % 3 == 0:
            # Numerical column
            data[column_names[i]] = np.random.randint(1, 100, num_rows)
        elif i % 3 == 1:
            # String column
            data[column_names[i]] = np.random.choice(
                ["apple", "banana", "cherry"], num_rows
            )
        else:
            # Mixed column (string + number)
            data[column_names[i]] = np.random.choice(
                ["apple", 42, "banana", 99], num_rows
            )
    df = pd.DataFrame(data)
    return df


# Example usage
df = create_random_df(10, 5)  # Create DataFrame with 10 rows and 5 columns
ColumnEditor(df)  # Launch UI
