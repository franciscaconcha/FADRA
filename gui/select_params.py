import Tkinter as tk
from dataproc.timeseries import astrointerface
import dataproc as dp

class select_params_window(tk.Tk):
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
        #return "hi"

    def on_closing(self):
        import tkMessageBox as messagebox
        self.destroy()
        #return "hi"


def select_params(sci_data):
    input_data = dp.AstroDir(sci_data['Raw_path'])
    print(sci_data['Raw_path'])
    first_im = input_data.readdata()[0]
    # Call to dataproc's astrointerface to select target and references
    interface = astrointerface.AstroInterface(first_im)
    interface.execute()
    labels = interface.labels
    points = [(b, a) for a, b in interface.dynamic_points]

    import matplotlib
    matplotlib.use('TkAgg')

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

    # implement the default mpl key bindings
    from matplotlib.backend_bases import key_press_handler


    from matplotlib.figure import Figure

    root = tk.Tk()
    root.wm_title("Embedding in TK")

    f = Figure(figsize=(5, 4), dpi=100)
    a = f.add_subplot(111)
    plot_data = dp.subarray(first_im, points[0], 20)
    stamp_rad = 20
    p0 = stamp_rad
    p1 = stamp_rad
    p = [p0, p1]
    circ_aperture = matplotlib.pyplot.Circle(p, 5, color='g', fill=False)
    circ_sky_inner = matplotlib.pyplot.Circle(p, 10, color='r', fill=False)
    circ_sky_outer = matplotlib.pyplot.Circle(p, 15, color='r', fill=False)
    a.add_artist(circ_aperture)
    a.add_artist(circ_sky_inner)
    a.add_artist(circ_sky_outer)
    a.imshow(plot_data, cmap = matplotlib.pyplot.get_cmap('gray'))

    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    stamp_rad_frame = tk.Frame(root)
    stamp_rad_frame.pack(side = tk.TOP)
    stamp_rad_label = tk.Label(stamp_rad_frame, text = "Stamp radius: ")
    stamp_rad_variable = tk.IntVar()
    stamp_rad_entry = tk.Entry(stamp_rad_frame, textvariable = stamp_rad_variable)
    stamp_rad_variable.set(50)
    stamp_rad_label.pack(side = tk.LEFT)
    stamp_rad_entry.pack(side = tk.LEFT)

    aperture_frame = tk.Frame(root)
    aperture_frame.pack(side = tk.TOP)
    aperture_label = tk.Label(aperture_frame, text = "Aperture: ")
    aperture_variable = tk.IntVar()
    aperture_entry = tk.Entry(aperture_frame, textvariable = aperture_variable)
    aperture_variable.set(10)
    aperture_label.pack(side = tk.LEFT)
    aperture_entry.pack(side = tk.LEFT)

    sky_inner_frame = tk.Frame(root)
    sky_inner_frame.pack(side = tk.TOP)
    sky_inner_label = tk.Label(sky_inner_frame, text = "Inner sky radius: ")
    sky_inner_variable = tk.IntVar()
    sky_inner_entry = tk.Entry(sky_inner_frame, textvariable = sky_inner_variable)
    sky_inner_variable.set(15)
    sky_inner_label.pack(side = tk.LEFT)
    sky_inner_entry.pack(side = tk.LEFT)

    sky_outer_frame = tk.Frame(root)
    sky_outer_frame.pack(side = tk.TOP)
    sky_outer_label = tk.Label(sky_outer_frame, text = "Outer sky radius: ")
    sky_outer_variable = tk.IntVar()
    sky_outer_entry = tk.Entry(sky_outer_frame, textvariable = sky_outer_variable)
    sky_outer_variable.set(20)
    sky_outer_label.pack(side = tk.LEFT)
    sky_outer_entry.pack(side = tk.LEFT)

    button = tk.Button(master=root, text='UPDATE', command = lambda: update_params(int(stamp_rad_entry.get()),
                                                                                   int(aperture_entry.get()),
                                                                                   int(sky_inner_entry.get()),
                                                                                   int(sky_outer_entry.get()),
                                                                                   a))
    button.pack(side=tk.BOTTOM)

    def update_params(stamp_rad, ap, sky_inner, sky_outer, a):
        plot_data = dp.subarray(first_im, points[0], stamp_rad)
        p0 = stamp_rad
        p1 = stamp_rad
        p = [p0, p1]
        circ_aperture = matplotlib.pyplot.Circle(p, ap, color='g', fill=False)
        circ_sky_inner = matplotlib.pyplot.Circle(p, sky_inner, color='r', fill=False)
        circ_sky_outer = matplotlib.pyplot.Circle(p, sky_outer, color='r', fill=False)
        a.clear()
        a.imshow(plot_data, cmap = matplotlib.pyplot.get_cmap('gray'))
        a.add_artist(circ_aperture)
        a.add_artist(circ_sky_inner)
        a.add_artist(circ_sky_outer)



    """go=True

    if(t==0):
		fits=pf.open(slist[0])
		tdata=fits[0].data
		t0=fits[0].header['MJD-OBS']
		fits.close()
	else:
		t0=t

	sdata=subarray(tdata,y,x,rad,path)
	ny,nx=centroide(sdata)
	data=subarray(sdata,ny,nx,rad,path)

	pl.ion()

	fig=pl.figure(figsize=(15, 15), dpi=40)

	ax1=fig.add_axes([0.3,0.55,0.4,0.4]) #left,bottom,width,height
	ax2=fig.add_axes([0.1,0.1,0.8,0.4])

	l,u=zscale(data)
	ax1.imshow(data,cmap=pl.cm.Greys_r)
	ax1.grid(True,color='red')

	y,x=sp.mgrid[0:2*rad+1,0:2*rad+1]
	d=sp.sqrt((y-ny)**2 + (x-nx)**2)
	ax2.plot(d,data,'k+')
	ax2.grid(True)
	xtext2=ax2.set_xlabel('pixels')
	ytext2=ax2.set_ylabel('flux')
	sky2=0

	while(go==True):

		try:
			ap1,ap2,sky1,sky2=input("Insert APERTURE 1,APERTURE 2,SKY-INNER,SKY-OUTER:")
			#getaperture(aperture,sky1,sky2,list,y,x,)

		except ValueError:
			print "Please insert APERTURE,SKY-INNER,SKY-OUTER\n"


		if(ap2==0):
			ap2=ap1

		ax1.clear()
		ax2.clear()

		#Primer Grafico: Imagen
		l,u=zscale(data)
		ax1.imshow(data,cmap=pl.cm.Greys_r)
		ax1.grid(True,color='red')

		circlea1=pl.Circle((ny,nx),int(ap1),color='g',fill=False) #aperture1
		circlea2=pl.Circle((ny,nx),int(ap2),color='g',fill=False) #aperture2
		circle2=pl.Circle((ny,nx),int(sky1),color='r',fill=False) #inner sky
		circle3=pl.Circle((ny,nx),int(sky2),color='r',fill=False) #outer sky
		ax1.add_artist(circlea1)
		ax1.add_artist(circlea2)
		ax1.add_artist(circle2)
		ax1.add_artist(circle3)
		xtext1=ax1.set_xlabel('pixels')
		ytext1=ax1.set_ylabel('pixels')

		#Segundo Grafico: Perfil radial
		y,x=sp.mgrid[0:2*rad+1,0:2*rad+1]
		d=sp.sqrt((y-ny)**2 + (x-nx)**2)
		ax2.plot(d,data,'k+')
		pl.vlines(ap1,data.min(),data.max(),color='g')
		pl.vlines(ap2,data.min(),data.max(),color='g')
		pl.vlines(sky1,data.min(),data.max(),color='r')
		pl.vlines(sky2,data.min(),data.max(),color='r')
		#ax2.grid(True)
		xtext2=ax2.set_xlabel('pixels')
		ytext2=ax2.set_ylabel('flux')

		isok=raw_input("Ok?: (y/n) ")

		if isok=="y":
			go=False
			#many=raw_input("Aperture range?: (y/n) ")
			#if many=="y":
		#		amin,amax=input("Insert Aperture range MIN,MAX: ")
		#		return aperture,[sky1,sky2],amin,amax

	return range(ap1,ap2+1),[sky1,sky2],t0#,None,None#,amin,amax"""