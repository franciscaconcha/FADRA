import Tkinter as tk
from tkFileDialog import askdirectory
import tkMessageBox as messagebox
import pickle

class simpleapp_tk(tk.Tk):

    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.open_dirs = []
        self.initialize()

    def initialize(self):
        self.grid()
        self.dir_list = tk.Listbox(font=("Arial", 12))
        #self.dir_list.pack()
        self.dir_list.grid(row = 0, column = 0, columnspan = 4, rowspan = 5, sticky = "W")
        self.dir_list.insert(tk.END, "No open AstroDirs")
        self.get_ts_button = tk.Button(self.parent, text = u"Obtain light curve",
                                       state=tk.DISABLED, command = self.get_lightcurve)
        self.get_ts_button.grid(row = 0, column = 4, sticky = "W", padx = (5, 5))
        self.view_ts_button = tk.Button(self.parent, text = u"View light curve",
                                        state=tk.DISABLED, command = self.hello)
        self.view_ts_button.grid(row = 1, column = 4, sticky = "W", padx = (5, 5))
        self.save_ad_button = tk.Button(self.parent, text = u"Save AstroDir",
                                          state=tk.DISABLED, command = self.remove_astrodir)
        self.save_ad_button.grid(row = 2, column = 4, sticky = "W", padx = (5, 5))
        self.remove_ad_button = tk.Button(self.parent, text = u"Remove AstroDir",
                                          state=tk.DISABLED, command = self.remove_astrodir)
        self.remove_ad_button.grid(row = 3, column = 4, sticky = "W", padx = (5, 5))

        def load_data_window():
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
            #global masters_given_var
            #masters_given_var = tk.IntVar()
            #checkbox_mg = tk.Checkbutton(self.t, text = "Masters ready (set above)", variable = masters_given_var,
            #                             onvalue = 1, offvalue = 0)
            #checkbox_mg.grid(row = 8 + i, column = 0, sticky = "W")

            #global combine_var
            #combine_var = tk.IntVar()
            #checkbox_cm = tk.Checkbutton(self.t, text = "Combine masters using: ", variable = combine_var)
            #checkbox_cm.grid(row = 9 + i, column = 0, sticky = "W")

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


        menubar = tk.Menu()

        # create a pulldown menu, and add it to the menu bar
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label = "Load existing AstroDir", command = self.load_astrodir)
        filemenu.add_command(label = "Create new AstroDir", command = load_data_window)
        filemenu.add_separator()
        filemenu.add_command(label = "Save current AstoDir", command = self.save_astrodir)
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # create more pulldown menus
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Combine data", command = self.hello)
        editmenu.add_command(label="Apply filter", command = self.hello)
        menubar.add_cascade(label="Edit", menu=editmenu)

        helpmenu = tk.Menu(menubar, tearoff = 0)
        helpmenu.add_command(label="About", command = self.hello)
        menubar.add_cascade(label="Help", menu = helpmenu)
        self.config(menu = menubar)

        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)  # horizontal, vertical
        self.update()
        self.geometry("300x300")

    def OnButtonClick(self):
        self.labelVariable.set( self.entryVariable.get()+" (You clicked the button)" )

    def OnPressEnter(self,event):
        self.labelVariable.set( self.entryVariable.get()+" (You pressed ENTER)" )

    def get_BIASfilepath(self):
        global BIAS_entryVariable
        name = askdirectory()
        BIAS_entryVariable.set(name)

    def get_Darkfilepath(self):
        global Dark_entryVariable
        name = askdirectory()
        Dark_entryVariable.set(name)

    def get_Flatfilepath(self):
        global Flat_entryVariable
        name = askdirectory()
        Flat_entryVariable.set(name)

    def get_Rawfilepath(self):
        global Raw_entryVariable
        name = askdirectory()
        Raw_entryVariable.set(name)

    def get_Savefilepath(self):
        global Save_entryVariable
        name = askdirectory()
        Save_entryVariable.set(name)

    def create_astrodir(self, bias_path, dark_path, flat_path, raw_path, save_path,
                        combine, combine_mode, dir_name):
        new_astrodir = {}
        new_astrodir['BIAS_path'] = bias_path
        new_astrodir['Dark_path'] = dark_path
        new_astrodir['Flat_path'] = flat_path
        new_astrodir['Raw_path'] = raw_path
        new_astrodir['Save_path'] = save_path
        new_astrodir['Combine'] = combine
        new_astrodir['Combine_mode'] = combine_mode
        new_astrodir['AstroDir_Name'] = dir_name
        new_astrodir['TS'] = None
        self.open_dirs.append(new_astrodir)
        self.reload_astrodirs()
        self.t.destroy()

    def load_astrodir(self):
        name = askdirectory()
        with open(name, 'rb') as handle:
            b = pickle.loads(handle.read())
        self.open_dirs.append(b)

    def reload_astrodirs(self):
        self.dir_list.delete(0, tk.END)
        for ad in self.open_dirs:
            self.dir_list.insert(tk.END, ad['AstroDir_Name'])
        #self.view_ts_button['state'] = tk.NORMAL
        self.get_ts_button['state'] = tk.NORMAL
        self.remove_ad_button['state'] = tk.NORMAL
        self.save_ad_button['state'] = tk.NORMAL

    def save_astrodir(self):
        pass
        #with open('file.txt', 'wb') as handle:
        #    pickle.dump(a, handle)

    def view_lightcurve(self):
        curr_ad_name = self.dir_list.get(self.dir_list.curselection())
        curr_ad = next((item for item in self.open_dirs if item['AstroDir_Name'] == curr_ad_name), None)
        if curr_ad['TS'] == None:
            messagebox.showinfo("Light curve for this AstroDir has not been calculated yet!")
        else:
            from src import TimeSeries
            curr_ad['TS'].plot()

    def get_lightcurve(self):
        curr_ad_name = self.dir_list.get(self.dir_list.curselection())
        curr_ad = next((item for item in self.open_dirs if item['AstroDir_Name'] == curr_ad_name), None)
        import select_params as sp
        s = sp.select_params(curr_ad)
        #from src import get_stamps
        #curr_ad['TS'] = get_stamps.photometry(curr_ad['Raw_path'], curr_ad['BIAS_path'], curr_ad['Dark_path'],
         #                                     curr_ad['Flat_path'], coords, ap, stamp_rad, sky, deg=deg, ron=ron,
         #                                     gpu=gpu)

    def remove_astrodir(self):
        import tkMessageBox as messagebox
        if messagebox.askokcancel("Quit", "Are you sure you want to remove this AstroDir from the list?"
                                          "(If the AstroDir is saved, it won't be deleted from disk)"):
            self.destroy()

    def hello(self):
        pass

    def on_closing(self):
        import tkMessageBox as messagebox
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.destroy()

if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('FADRA')
    app.mainloop()
