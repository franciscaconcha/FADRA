    #!/usr/local/bin/cl.e -f

    set imdir = "HDR$pixels/" 		# default environment
    logver = "IRAF V2.11 May 1997"	# needed for IMAGES package

    images				# load needed packages
    imfilter
    {
	# Execute the command.
	printf ("imfilter.boxcar %s\n", args) | cl()

        logout				# shut down
    }
