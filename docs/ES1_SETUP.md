**Read this ONLY IF you have a IWR1642BOOST ES1.0 board.**

Our board is IWR1642BOOST ES1.0. The revision number ES1.0 means that we will not be able to run higher version of the SDK and CCS.
CCS (Code Composer Studio) is a IDE for the board. To my understanding so far we need to burn codes into the board and use this studio
to control the mmWave. Version 7.4 (newest) is confirmed not working.

Below is a brief introduction of setup, will add more as we explore deeper:
  1. Install SDK 1.2. The naming is very confusing but SDK 1.2 is SDK 1.02. Download it from this [TI link](http://software-dl.ti.com/ra-processors/esd/MMWAVE-SDK/01_02_00_05/index_FDS.html)
     (confirmed working) or this [GitHub repo](https://github.com/anirudh-ramesh/TI-mmWave-SDK) (not sure if it is working but convenient to find).
  2. Download CCS 7.1 from [here](http://processors.wiki.ti.com/index.php/Download_CCS#Code_Composer_Studio_Version_7_Downloads). Upon finishing
     installation, it will ask you to update resource explorer. Follow its instruction to update it (it will redirect you to this [link](http://processors.wiki.ti.com/index.php/Download_CCS#Code_Composer_Studio_Version_7_Downloads)).
  3. After finishing the update, the out-of-the-box industrial toolbox will not work for ES1.0, so we need to use an older version of it. The
     official recommended version is 2.3.0, however it is nowhere to be found, so we need to install 2.2.0. Click the *Home* button on the top
     right corner, scroll down until you see the tile for industrial toolbox, click the expandable list, select 2.2.0 and install. Note that if you see
     the industrial toolbox is still 3.3.0 after restarting CCS, you probably need to reselect 2.3.0 in the home.
  4. Follow the given PDF file until you reach **import MSS and DSS to IDE**. If you encounter the problem as `Error importing project. Device-id "Cortex R.IWR1642" is not recognized!`.
     You need to install the [mmWave Radar Device Support](https://e2e.ti.com/support/sensors/f/1023/t/730398) in order to recognize the device. The procedure
     is similar to updating the TIREX. Go to Help -> Install new Software -> Selectr "--All Available Site--" in the **Work with** drop-down menu->
     enter `mmwave` in the textbox below -> select **mmwave radar device support** (I installed version 1.6.2) to install.
  5. Follow the PDF until rebuilding the project in step 3. If you follow my procedure you probably won't find the `mmw` project to rebuild after
     rebuilding dss. In this case, just rebuild mss, it should be the same thing.
