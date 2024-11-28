1. Connect to the CARC server. If you are not on USC secure you will need to download a VPN client
in order to connect to the USC server: https://www.carc.usc.edu/user-guides/quick-start-guides/anyconnect-vpn-setup

2. To connect to the carc server, in terminal run: ssh userid@discovery.usc.edu and enter your password. 
User id is your usc id (example: daxner@discover.usc.edu).

3. You will need to use scp to upload the folder to the carc server. scp -r folder_directory userid@discover.usc.edu:~. Alternatively, setup the ssh connection
through ssh extension(makes it a lot easier to edit files on the carc server to, heavily recommended).

4. To run the training there are .sh files included in the folder. To start the program execute 'sbatch finetun_a40.sh' or 'sbatch finetun_a100.sh'. Once the program starts
a slurm file with the job id will generate in the file(.out extension), this will show the output of the program. Either look at the file in vs code or run `cat filename.out`. 
You can see if the job has started with `squeue $USER` and stop the job with `scancel jobID`. The a40 file specifies the a40 GPU, a100 specices a100 GPU. Seems that a100's are in higher demand
so you might have to wait longer for the a100 to be available and start. 

5. Once training is done a folder named outputs will be generated, this is the saved weights once training is done.

6. To edit the queries that will be used look at process_data.py

# TODO: import the saved weights for second and 3rd stage training