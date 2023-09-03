# Deployment Instructions

Note: all make files are in **/setup** but preferred way of running is from run folder - it has some preconfigured scripts

  1. Navigate to the **/run** directory
  2. Run bash **script_name**
  3. There should be one to launch as well as shutdown. The launch automatically shuts down the server if it was running.
  
If you want to run make directly - got to **/setup**

  1. Run `make install-tools` to setup dependencies
  2. Run `make launch-[component]` to launch the application
  3. Run `make shutdown-[component]` to shutdown the application

Once deployed go to - http://localhost:5005 (This is the ui homepage). 
