import Tkinter as tk


class load_data_window(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.open_dirs = []
        self.initialize()

    def initialize(self):
            self.t = tk.Toplevel()

            # File paths
            i = 6
            title_label = tk.Label(self.t, text = "Choose file paths", font = ("Verdana", 10, "bold"))
            title_label.grid(row = 0, column = 0, columnspan = 2, sticky = "W")
            instructions_label = tk.Label(self.t, text="Choose path for each kind of file.\nIf you don't have the files "
                                                  "in separate folders, choose your path on\nthe first text entry (BIAS) "
                                                  "and check the \'Not separated\' box.\nIf you already have master "
                                                  "files ready, select them instead of the path.", justify = "left")
            instructions_label.grid(row = 1, column = 0, columnspan = 3, rowspan = i)

            # BIAS
            BIAS_label = tk.Label(self.t, text = "BIAS files path: ")
            BIAS_label.grid(row = 2 + i, column = 0, sticky = "E", pady = (10, 10))
            global BIAS_entryVariable
            BIAS_entryVariable = tk.StringVar()
            BIAS_entry = tk.Entry(self.t, textvariable = BIAS_entryVariable)
            BIAS_entry.grid(row = 2 + i, column = 1, columnspan = 2, sticky = "EW")
            BIAS_entryVariable.set(u"Select path")
            BIAS_button = tk.Button(self.t, text = u"Select...", command = self.get_BIASfilepath)
            BIAS_button.grid(row = 2 + i, column = 3, sticky = "W", padx = (5, 5))

            # "Not separated" check box
            not_separated = tk.IntVar()
            checkbox = tk.Checkbutton(self.t, text = "Not separated", variable = not_separated)
            checkbox.grid(row = 2 + i, column = 4, sticky = "W")

            # Dark
            Dark_label = tk.Label(self.t, text = "Dark files path: ")
            Dark_label.grid(row = 3 + i, column = 0, sticky = "E", pady = (10, 10))
            global Dark_entryVariable
            Dark_entryVariable = tk.StringVar()
            Dark_entry = tk.Entry(self.t, textvariable = Dark_entryVariable)
            Dark_entry.grid(row = 3 + i, column = 1, columnspan = 2, sticky = "EW")
            Dark_entryVariable.set(u"Select path")
            Dark_button = tk.Button(self.t, text = u"Select...", command = self.get_Darkfilepath)
            Dark_button.grid(row = 3 + i, column = 3, sticky = "W", padx = (5, 5))

            # Flat
            Flat_label = tk.Label(self.t, text = "Flat files path: ")
            Flat_label.grid(row = 4 + i, column = 0, sticky = "E", pady = (10, 10))
            global Flat_entryVariable
            Flat_entryVariable = tk.StringVar()
            Flat_entry = tk.Entry(self.t, textvariable = Flat_entryVariable)
            Flat_entry.grid(row = 4 + i, column = 1, columnspan = 2, sticky = "EW")
            Flat_entryVariable.set(u"Select path")
            Flat_button = tk.Button(self.t, text = u"Select...", command = self.get_Flatfilepath)
            Flat_button.grid(row = 4 + i, column = 3, sticky = "W", padx = (5, 5))

            # Raw
            Raw_label = tk.Label(self.t, text = "Raw science files path: ")
            Raw_label.grid(row = 5 + i, column = 0, sticky = "E", pady = (10, 10))
            global Raw_entryVariable
            Raw_entryVariable = tk.StringVar()
            Raw_entry = tk.Entry(self.t, textvariable = Raw_entryVariable)
            Raw_entry.grid(row = 5 + i, column = 1, columnspan = 2, sticky = "EW")
            Raw_entryVariable.set(u"Select path")
            Raw_button = tk.Button(self.t, text = u"Select...", command = self.get_Rawfilepath)
            Raw_button.grid(row = 5 + i, column = 3, sticky = "W", padx = (5, 5))

            # Save path
            Save_label = tk.Label(self.t, text = "Path to save files: ")
            Save_label.grid(row = 6 + i, column = 0, sticky = "E", pady = (10, 10))
            global Save_entryVariable
            Save_entryVariable = tk.StringVar()
            Save_entry = tk.Entry(self.t, textvariable = Save_entryVariable)
            Save_entry.grid(row = 6 + i, column = 1, columnspan = 2, sticky = "EW")
            Save_entryVariable.set(u"Select path")
            Save_button = tk.Button(self.t, text = u"Select...", command = self.get_Savefilepath)
            Save_button.grid(row = 6 + i, column = 3, sticky = "W", padx = (5, 5))

            # Master files
            title2_label = tk.Label(self.t, text = "Choose combine method for Master files", font = ("Verdana", 10, "bold"))
            title2_label.grid(row = 7 + i, column = 0, columnspan = 2, sticky = "W")

            # Master combine checkbox
            """global masters_given_var
            masters_given_var = tk.IntVar()
            checkbox_mg = tk.Checkbutton(self.t, text = "Masters ready (set above)", variable = masters_given_var,
                                         onvalue = 1, offvalue = 0)
            checkbox_mg.grid(row = 8 + i, column = 0, sticky = "W")

            global combine_var
            combine_var = tk.IntVar()
            checkbox_cm = tk.Checkbutton(self.t, text = "Combine masters using: ", variable = combine_var)
            checkbox_cm.grid(row = 9 + i, column = 0, sticky = "W")"""

            combine_var = tk.IntVar()
            radiobutton_mg = tk.Radiobutton(self.t, text = "Masters ready (set above)", variable = combine_var,
                                             value = 0)
            radiobutton_mg.grid(row = 8 + i, column = 0, sticky = "W")

            radiobutton_comb = tk.Radiobutton(self.t, text = "Combine masters using: ", variable = combine_var,
                                             value = 1)
            radiobutton_comb.grid(row = 9 + i, column = 0, sticky = "W")

            global combine_mode_var
            combine_mode_var = tk.StringVar()
            combine_choices = ["Median", "Mean"]
            combine_option = tk.OptionMenu(self.t, combine_mode_var, *combine_choices)
            combine_option.grid(row = 9 + i, column = 1, sticky = "W")

            # Give AstroDir a name
            title3_label = tk.Label(self.t, text = "AstroDir options", font = ("Verdana", 10, "bold"))
            title3_label.grid(row = 10 + i, column = 0, columnspan = 2, sticky = "W")
            DirName_label = tk.Label(self.t, text = "Name of this AstroDir: ")
            DirName_label.grid(row = 11 + i, column = 0, sticky = "W")
            #global DirName_entryVariable
            DirName_entryVariable = tk.StringVar()
            DirName_entry = tk.Entry(self.t, textvariable = DirName_entryVariable)
            DirName_entry.grid(row = 11 + i, column = 1, sticky = "W")

            # "Go" Button
            go_button = tk.Button(self.t, text = u"Create AstroDir",
                                  command = lambda: self.create_astrodir(BIAS_entryVariable.get(),
                                                                         Dark_entryVariable.get(),
                                                                         Flat_entryVariable.get(),
                                                                         Raw_entryVariable.get(),
                                                                         Save_entryVariable.get(),
                                                                         combine_var.get(),
                                                                         combine_mode_var.get(),
                                                                         DirName_entryVariable.get()))
            go_button.grid(row = 12 + i, column = 1, sticky = "W", pady=(20, 20))

            #t.grid_columnconfigure(0, weight=1)
            self.t.resizable(True, True)  # horizontal, vertical
            self.t.update()
            self.t.geometry("650x450")