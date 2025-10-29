# Run Demo on Custom Data
In this tutorial we introduce the demo of OnePose running with data captured
with our **OnePose Cap** application available for iOS device. 
You can collect your own data by following the steps below, or directly use our [example demo data](https://drive.google.com/file/d/15SfRd_h5DPb6FkZzcz7lcIKV0TmgGei9/view?usp=sharing) to skip Step 1 and quickly try out our model.

### Step 1: Capture the mapping sequence and the test sequence with OnePose Cap. 
#### The app (IOS only) is available at [Appstore](https://apps.apple.com/cn/app/onepose-capture/id6447052065?l=en-GB).

### Step 2: Organize the file structure of collected sequences
1. Download the captured mapping sequence and the test sequence to the PC by the provided url in app.
2. Rename the **annotate** and **test** sequences directories to ``your_obj_name-annotate`` and `your_obj_name-test` respectively and organize the data as the follow structure:
    ```
    |--- /your/path/to/scanned_data
    |       |--- your_obj_name
    |       |       |---your_obj_name-annotate
    |       |       |---your_obj_name-test
    ```
   Refer to the [demo data](https://drive.google.com/file/d/15SfRd_h5DPb6FkZzcz7lcIKV0TmgGei9/view?usp=sharing) as an example.
3. Link the collected data to the project directory
    ```shell
    REPO_ROOT=/path/to/OnePose
    ln -s /path/to/scanned_data $REPO_ROOT/data/demo
    ```
   
Now the data is prepared!

### Step 3: Run OnePose with collected data
Execute the following commands, and a demo video naming `demo_video.mp4` will be saved in the folder of the test sequence.
```shell
REPO_ROOT=/path/to/OnePose
OBJ_NAME=your_obj_name

cd $REPO_ROOT
conda activate oneposeplus

bash scripts/demo_pipeline.sh $OBJ_NAME

```
