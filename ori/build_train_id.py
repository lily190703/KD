import json
import numpy as np
file_train = '/mnt/sda1/wanghaoran/lcxre/wyj/original/code/dataset_splits/ori/something-something-v2-train.json'
# file_train_id = '/root/wanghaoran/wyj/ori_moreVP/code/dataset_splits/something-something-v2-train.json'


train_id_new = []
with open(file_train) as f:
	class_list = json.load(f)
for class_dict in class_list:
	if class_dict['template'] == "Putting [something] that can't roll onto a slanted surface, so it slides down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Lifting a surface with [something] on it until it starts sliding down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Letting [something] roll down a slanted surface":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Letting [something] roll up a slanted surface, so it rolls back down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Putting [number of] [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Taking [something] out of [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Putting [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Putting [something] next to [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Putting [something] behind [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Taking [something] from [somewhere]":
		train_id_new.append(str(class_dict['id']))



	if class_dict['template'] == "Tipping [something] with [something in it] over, so [something in it] falls out":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Tipping [something] over":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pulling [something] out of [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pulling [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pulling [something] from left to right":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pushing [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pushing [something] so that it falls off the table":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pushing [something] off of [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Moving [something] down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Hitting [something] with [something]":
		train_id_new.append(str(class_dict['id']))



	if class_dict['template'] == "Poking [something] so that it falls over":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Holding [something] in front of [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Holding [something] behind [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Holding [something] over [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Trying to pour [something] into [something], but missing so it spills next to it":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pouring [something] out of [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pouring [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pouring [something] into [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Pouring [something] into [something] until it overflows":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Spilling [something] next to [something]":
		train_id_new.append(str(class_dict['id']))



	if class_dict['template'] == "Spilling [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Spilling [something] behind [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Moving [something] across a surface until it falls down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Lifting [something] up completely, then letting it drop down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Lifting [something] up completely without letting it drop down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Dropping [something] into [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Dropping [something] onto [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Dropping [something] next to [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Dropping [something] in front of [something]":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Dropping [something] behind [something]":
		train_id_new.append(str(class_dict['id']))



	if class_dict['template'] == 'Moving [something] up':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Putting [something] onto [something else that cannot support it] so it falls down':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Putting [something] on the edge of [something] so it is not supported and falls down':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Lifting up one end of [something], then letting it drop down':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == "Putting [something] onto a slanted surface but it doesn't glide down":
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Putting [something] into [something]':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Holding [something] next to [something]':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Pulling [something] from right to left':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Putting [something] on a flat surface without letting it roll':
		train_id_new.append(str(class_dict['id']))

	if class_dict['template'] == 'Pretending to pour [something] out of [something], but [something] is empty':
		train_id_new.append(str(class_dict['id']))


print(train_id_new)
# np.savetxt('out.txt', train_id_new)








