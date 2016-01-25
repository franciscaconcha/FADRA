import Tkinter as tk


class simpleapp_tk(tk.Tk):

    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()

        def hello():
            print "hello!"

        def load_data_window():
            t = tk.Toplevel()

            # File paths
            i = 6
            title_label = tk.Label(t, text = "Choose file paths", font = ("Verdana", 10, "bold"))
            title_label.grid(row = 0, column = 0, columnspan = 2, sticky = "W")
            instructions_label = tk.Label(t, text="Choose path for each kind of file.\nIf you don't have the files "
                                                  "in separate folders, choose your path on\nthe first text entry (BIAS) "
                                                  "and check the \'Not separated\' box.\nIf you already have master "
                                                  "files ready, select them instead of the path.", justify = "left")
            instructions_label.grid(row = 1, column = 0, columnspan = 3, rowspan = i)

            # BIAS
            BIAS_label = tk.Label(t, text = "BIAS files path: ")
            BIAS_label.grid(row = 2 + i, column = 0, sticky = "E", pady=(10, 10))
            BIAS_entryVariable = tk.StringVar()
            BIAS_entry = tk.Entry(t, textvariable = BIAS_entryVariable)
            BIAS_entry.grid(row = 2 + i, column = 1, columnspan = 2, sticky = "EW")
            BIAS_entryVariable.set(u"Select path")
            BIAS_button = tk.Button(t, text = u"Select...")
            BIAS_button.grid(row = 2 + i, column = 3, sticky = "W", padx=(5, 5))

            # "Not separated" check box
            not_separated = tk.IntVar()
            checkbox = tk.Checkbutton(t, text = "Not separated", variable = not_separated)
            checkbox.grid(row = 2 + i, column = 4, sticky = "W")

            # Dark
            Dark_label = tk.Label(t, text = "Dark files path: ")
            Dark_label.grid(row = 3 + i, column = 0, sticky = "E", pady=(10, 10))
            Dark_entryVariable = tk.StringVar()
            Dark_entry = tk.Entry(t, textvariable = Dark_entryVariable)
            Dark_entry.grid(row = 3 + i, column = 1, columnspan = 2, sticky = "EW")
            Dark_entryVariable.set(u"Select path")
            Dark_button = tk.Button(t, text = u"Select...")
            Dark_button.grid(row = 3 + i, column = 3, sticky = "W", padx=(5, 5))

            # Flat
            Flat_label = tk.Label(t, text = "Flat files path: ")
            Flat_label.grid(row = 4 + i, column = 0, sticky = "E", pady=(10, 10))
            Flat_entryVariable = tk.StringVar()
            Flat_entry = tk.Entry(t, textvariable = Flat_entryVariable)
            Flat_entry.grid(row = 4 + i, column = 1, columnspan = 2, sticky = "EW")
            Flat_entryVariable.set(u"Select path")
            Flat_button = tk.Button(t, text = u"Select...")
            Flat_button.grid(row = 4 + i, column = 3, sticky = "W", padx=(5, 5))

            # Raw
            Raw_label = tk.Label(t, text = "Raw science files path: ")
            Raw_label.grid(row = 5 + i, column = 0, sticky = "E", pady=(10, 10))
            Raw_entryVariable = tk.StringVar()
            Raw_entry = tk.Entry(t, textvariable = Raw_entryVariable)
            Raw_entry.grid(row = 5 + i, column = 1, columnspan = 2, sticky = "EW")
            Raw_entryVariable.set(u"Select path")
            Raw_button = tk.Button(t, text = u"Select...")
            Raw_button.grid(row = 5 + i, column = 3, sticky = "W", padx=(5, 5))

            # Save path
            Save_label = tk.Label(t, text = "Path to save files: ")
            Save_label.grid(row = 6 + i, column = 0, sticky = "E", pady=(10, 10))
            Save_entryVariable = tk.StringVar()
            Save_entry = tk.Entry(t, textvariable = Save_entryVariable)
            Save_entry.grid(row = 6 + i, column = 1, columnspan = 2, sticky = "EW")
            Save_entryVariable.set(u"Select path")
            Save_button = tk.Button(t, text = u"Select...")
            Save_button.grid(row = 6 + i, column = 3, sticky = "W", padx=(5, 5))

            # Master files
            title2_label = tk.Label(t, text = "Choose combine method for Master files", font = ("Verdana", 10, "bold"))
            title2_label.grid(row = 7 + i, column = 0, columnspan = 2, sticky = "W")

            # Master combine checkbox
            masters_given = tk.IntVar()
            checkbox_mg = tk.Checkbutton(t, text="Masters ready (set above)", variable=masters_given)
            checkbox_mg.grid(row = 8 + i, column = 0, sticky = "W")
            combine_masters = tk.IntVar()
            checkbox_cm = tk.Checkbutton(t, text="Combine masters using: ", variable=combine_masters)
            checkbox_cm.grid(row = 9 + i, column = 0, sticky = "W")

            combine_mode = tk.StringVar()
            combine_choices = ["Median", "Mean"]
            combine_option = tk.OptionMenu(t, combine_mode, *combine_choices)
            combine_option.grid(row = 9 + i, column = 1, sticky = "W")

            # Give AstroDir a name
            title3_label = tk.Label(t, text = "AstroDir options", font = ("Verdana", 10, "bold"))
            title3_label.grid(row = 10 + i, column = 0, columnspan = 2, sticky = "W")
            DirName_label = tk.Label(t, text = "Name of this AstroDir: ")
            DirName_label.grid(row = 11 + i, column = 0, sticky = "W")
            DirName_entryVariable = tk.StringVar()
            DirName_entry = tk.Entry(t, textvariable = DirName_entryVariable)
            DirName_entry.grid(row = 11 + i, column = 1, sticky = "W")

            # "Go" Button
            go_button = tk.Button(t, text=u"Create AstroDir")
            go_button.grid(row = 12 + i, column = 1, sticky = "W", pady=(20, 20))


            #t.grid_columnconfigure(0, weight=1)
            t.resizable(True, True)  # horizontal, vertical
            t.update()
            t.geometry("650x450")


        menubar = tk.Menu()

        # create a pulldown menu, and add it to the menu bar
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load existing AstroDir", command=hello)
        filemenu.add_command(label="Create new AstroDir", command=load_data_window)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # create more pulldown menus
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Combine data", command=hello)
        editmenu.add_command(label="Apply filter", command=hello)
        menubar.add_cascade(label="Edit", menu=editmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=hello)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)  # horizontal, vertical
        self.update()
        self.geometry("300x300")

    def OnButtonClick(self):
        self.labelVariable.set( self.entryVariable.get()+" (You clicked the button)" )

    def OnPressEnter(self,event):
        self.labelVariable.set( self.entryVariable.get()+" (You pressed ENTER)" )


if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('FRACTAL')
    app.mainloop()
