When using this algorithm for publications, please acknowledge the work presented in:
http://arxiv.org/abs/1603.00473

========================================================
INSTALLATION: 
	      - Download and unzip the MuSCADeT.zip file
	      - Open the MuSCADeT folder
	      - Install the python package:
	      		"python setup.py install"

========================================================
TESTS:
	In MuSCADeT, open the Examples file and execute:
	   	       "python Example_simple.py"

	Inputs provided in the Simu_simple folder:
	      -Cube.fits (fits file with input images)
	      -Simu_A.fits (mixing matrix used to generate the simulations)

	Results are written in the "Simu_simple" folder.
	After the execution, one can find the following files in the Simu_simple folder:
	      -Blue_residuals.fits (fits RGBcube with result from subtraction of red source from original band images)
	      -Colour_residuals.fits (fits RGBcube with result from subtraction of red and blue sources from original band images)
	      -Red_residuals.fits (fits RGBcube with result from subtraction of blue source from original band images)
	      -Estimated_A.fits (mixing matrix as estimated from spectral PCA)
	      -Sources_100.fits (fits cube with sources extracted by MuSCADeT: the direct product of our algorithm)


	All "Example_-.py" files can be used the same way. Each Simu_- folder contains the benchmark data or real images to be treated in the “Cube.fits” fits cube and the mixing coefficients used to generate the simulations in “Simu_A.fits”.
	"Example_refsdal.py" runs the algorithm on the real images of MACS J1149+2223 and writes the results in Simu_Refsdal_big

	Users may change the input parameters via the same files and are encouraged to use them for further applications.

========================================================
HOW TO USE:
	We provide an Example_For_User.py file, that users may fill in with their own inputs and where the commands are a bit more detailed than in the other examples.

MuSCADeT needs very little amount of inputs. Ideally, only the data are necessary. Since MuSCADeT needs the SEDs of the objects to separate to run, it is sometimes necessary though to provide a few more optional inputs to estimate the SEDs. There are several ways for MuSCADeT to get good SED estimation:

- Use your own SED: It is possible to provide MuSCADeT with your own SEDs. For instance, when dealing with a large survey, it possible to estimate the SEDs from a small sample of red and blue galaxies, save the corresponding SEDs and use them for the rest of the analysis. In general, this is a good option when dealing with a large number of objects to separate from a consistent set of observations. It is rather unadvised to use the SEDs and data a cube from different surveys.

- Have faith and let MuSCADeT perform the SED estimation on its own. Don’t touch anything and let the estimator do its job. It works in most cases, but shamefully fails if sources with different colours present a great overlap.

- Tweek it! Another option, which is less automated is to set the plot option to True. MuSCADeT will try to automatically estimate the SEDs but will show the results of the estimation in PCA space (fig. 2 in arxiv:1603.00473). The first plot should show the first 2 PCA components of all SEDs in the image. pixels with the same colour should form alignements in this plot. If the alignements are coloured in red and green on the plot (as in fig. 2 in arxiv:1603.00473), you can expect good separation. If they are not, then give values for optional variable alpha. the two values that one should give to alpha are the angles at which the PCA coefficients are aligned. Evaluate visually a rough estimation of these angles from the plot in PCA space, input these values in alpha, run the code again. If you leave the plot option to one, the same plots will appear, but the green and blue colourations will be aligned with the angles you provided. If they correspond to alignements of PCA coefficients, close the windows and let the code run.


My personal recipe for dealing with large survey:

Find an area in the survey where red and blue objects are clearly identified. Run MuSCADeT with the plot option set to one. Check that the SEDs are well estimated. If not, provide values for alpha and try again.
Once SEDs are well estimated, let the code run, check that the separation is well done, but if the area was chosen with almost no blending of red and blue sources, it should be trivial for MuSCADeT. At the end of the execution, have the mixing coefficients saved and then use them on the rest of your data.


Since the SED estimation is a crucial step in separating images, please do not hesitate emailing me support via github, or at: remy.joseph@epfl.ch.







