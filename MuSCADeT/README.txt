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




