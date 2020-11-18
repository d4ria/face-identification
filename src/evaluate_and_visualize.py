"""A script for metrics visualization."""
# import modules needed
from metrics import intersection_over_union as iou, area_ratio
import pandas as pd
import seaborn as sns
import pickle
sns.set()


# define row functions
def row_iou(row):
    res = iou((row['x_1_true'], row['y_1_true'], row['width_true'], row['height_true']),
              (row['x_1_pred'], row['y_1_pred'], row['width_pred'], row['height_pred']))
    return res


def row_area_ratio(row):
    ratio = area_ratio((row['x_1_true'], row['y_1_true'], row['width_true'], row['height_true']),
                       (row['x_1_pred'], row['y_1_pred'], row['width_pred'], row['height_pred']))
    return ratio

# read dataframes
true_boxes = pd.read_table('../annotations/list_bbox_celeba.txt',
                           delim_whitespace=True)
with open('../data/hog_bounding_boxes_dict.pkl', 'rb') as f:
    predicted_boxes = pd.DataFrame(pickle.load(f))
# merge dataframes
merged_boxes = true_boxes.merge(predicted_boxes, on='image_id',
                                suffixes=('_true', '_pred'))
# calculate metrics
merged_boxes['iou'] = merged_boxes.apply(lambda row: row_iou(row), axis=1)
merged_boxes['area_ratio'] = merged_boxes.apply(lambda row: row_area_ratio(row), axis=1)

# save visualizations
relation_plot = sns.scatterplot(data=merged_boxes, x="area_ratio", y="iou")
relation_plot.set(xlabel='Area Ratio', ylabel='Intersection over Union')
relation_fig = relation_plot.get_figure()
relation_fig.savefig('../data/iou_arearatio_relation.png')

iou_distribution = sns.displot(merged_boxes, x="iou", binwidth=0.05)
iou_distribution.set(xlabel='Intersection over Union', ylabel='Number of examples')
iou_fig = iou_distribution.fig
iou_fig.savefig('../data/iou_distribution.png')

area_ratio_dist = sns.displot(merged_boxes, x="area_ratio", binwidth=0.05)
area_ratio_dist.set(xlabel='Area Ratio', ylabel='Number of examples')
ratio_fig = area_ratio_dist.fig
ratio_fig.savefig('../data/area_ratio_distribution.png')

