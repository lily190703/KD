# IKD : Interaction Knowledge Distillation  

## The validated process of result for Knowledge Distillation

### Date Preparation

- Download the `something-something` database provided in the paper [The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/pdf/1706.04261.pdf).
- If network permission allowed,  you can also download the dataset on the [Baidu Netdisk]()

### Training & Testing 

- To demonstrate the effectiveness of the distillation part in the IKD method, we provide distillation experimental data and model results using a 10% labeled `something-something V2` dataset for testing and validation.

- To validate the result, run `./fusion_jisuan.py`:

  ```
  python fusion_jisuan.py
  ```


- To test the model for the knowledge distillation, run `./train_kd_test.py`:

  ```
  python train_kd_test.py --root_frames /path/to/Something-Else/20bn/
  						--json_data_val dataset_splits/sthv2_10/something-something-v2-validation.json
  						--json_file_labels dataset_splits/sthv2_10/labels.json
  						--tracked_boxes /path/to/bounding_box_annotations.json
  						--save_path /path/to/STHV2ALL/
  						--model_path ckpt/ck_p_10_e21_latest.pth.tar
  ```

If you have any questions or suggestions, feel free to contact me at my email: lily190703@gmail.com.