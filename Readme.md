# IKD : Interaction Knowledge Distillation  

## The validated process of result for Knowledge Distillation

### Date Preparation

- sth数据集训练所需的样本数据在STHV2ALL.tar文件中，解压文件到代码文件夹中，解压命令如下：

  ```linux
  tar -xf STHV2ALL.tar -C /path/to/code directory
  ```

### Training & Testing 

- 为了说明IKD方法中蒸馏部分的有效性，我们提供了使用10%有标签的sth数据集进行的蒸馏实验数据以及模型结果供审稿人测试验证。

- To validate the model, run `./train_kd_test.py`:

  ```
  python train_kd_test.py --logname experience_name --batch_size 72
  						--json_data_val ./dataset_splits/sthv2_10/something-something-v2-validation
  ```

  