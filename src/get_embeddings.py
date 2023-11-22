import os
import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from learning3d.models import PointNet
from learning3d.models import Classifier
from learning3d.data_utils import ClassificationData, ModelNet40Data


class MyPointNet(PointNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_embeddings(self, input_data):
        # Here we assume that the input_data has already been transferred to the appropriate device
        if self.input_shape == "bnc":
            input_data = input_data.permute(0, 2, 1)

        x = input_data
        for idx, layer in enumerate(self.layers[:-1]):  # Exclude the last ReLU
            x = layer(x)
            if idx == 4:
                break

        # If using global features, apply max pooling
        if self.global_feat:
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x


def test_one_epoch(device, model, test_loader, testset, save_dir='data'):
    model.eval()
    test_loss = 0.0
    pred = 0.0
    count = 0
    all_embeddings = []  # List to accumulate embeddings

    # Create the directory for saving embeddings and images if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            points, target = data
            target = target[:,0]

            points = points.to(device)
            target = target.to(device)

            output = model(points)
            loss_val = F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum')
            test_loss += loss_val.item()
            count += output.size(0)

            _, pred1 = output.max(dim=1)
            correct = (pred1 == target)
            pred += correct.sum().item()

            # Get the embeddings from the model and accumulate them
            embeddings = model.feature_model.get_embeddings(points)
            all_embeddings.append(embeddings.detach().cpu().numpy())

            
            # Save the point cloud image to a file
            points_np = points.detach().cpu().numpy()
            print(points_np.shape)
            # Combine all point clouds in the batch into one for saving
            batch_point_cloud = o3d.geometry.PointCloud()
            for p in points_np:
                single_point_cloud = o3d.geometry.PointCloud()
                single_point_cloud.points = o3d.utility.Vector3dVector(p[:, :3])  # Assuming x, y, z coordinates are the first three columns
                batch_point_cloud += single_point_cloud
            # Save the combined batch point cloud
            point_cloud_filename = os.path.join(save_dir, f"batch_point_cloud_{i}.ply")
            o3d.io.write_point_cloud(point_cloud_filename, batch_point_cloud)
            
            # # Save the point cloud image to a file
            # points_np = points.detach().cpu().numpy()
            # for j, p in enumerate(points_np):
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(p[:, :3])  # Assuming x, y, z coordinates are the first three columns
            #     point_cloud_filename = os.path.join(save_dir, f"point_cloud_{i}_{j}.ply")
            #     o3d.io.write_point_cloud(point_cloud_filename, point_cloud)

            # Optionally print out info
            print(f"Batch {i} - Loss: {loss_val.item()}")
            print("Ground Truth Label: ", testset.get_shape(target[0].item()))
            print("Predicted Label:    ", testset.get_shape(torch.argmax(output[0]).item()))

    # Concatenate and save all embeddings after the loop
    all_embeddings_np = np.concatenate(all_embeddings, axis=0)
    np.save(os.path.join(save_dir, "all_embeddings.npy"), all_embeddings_np)

    test_loss /= count
    accuracy = float(pred) / count

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return test_loss, accuracy

def save_point_cloud_image(point_cloud, filename):
    o3d.io.write_point_cloud(filename, point_cloud)




def test(args, model, test_loader, testset):
    test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader, testset, save_dir="data/")

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PointNet
	parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('--pretrained', default='learning3d/pretrained/exp_classifier/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
    args = options()
    args.dataset_path = os.path.join(os.getcwd(), 'ModelNet40', 'ModelNet40')
    
    testset = ClassificationData(ModelNet40Data(train=False))
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointNet Model.
    ptnet = MyPointNet(emb_dims=args.emb_dims, use_bn=True)
    model = Classifier(feature_model=ptnet)

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location=args.device))
    model.to(args.device)

    test(args, model, test_loader, testset)
