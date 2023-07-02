import torch
import numpy as np
import random
from scipy.spatial import distance
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import colorcet as cc
import matplotlib

matplotlib.use('Agg')


# from eval import Visualize_DTW_TimeStamps


class BatchGenerator:
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.list_of_examples = []
        self.index = 0
        self.gt = {}
        self.confidence_mask = {}

        # Load annotations
        # Where do we exactly use it?
        annotation_file_path = "/content/TimestampActionSeg/data/gtea_annotation_all.npy"
        self.random_index = np.load(annotation_file_path, allow_pickle=True).item()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        self.generate_confidence_mask()

    def generate_confidence_mask(self):
        for vid in self.list_of_examples:
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            self.gt[vid] = classes
            num_frames = classes.shape[0]
            # is "random_index" the annotated frame?
            random_idx = self.random_index[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_idx) - 1):
                left_mask[int(classes[random_idx[j]]), random_idx[j]:random_idx[j + 1]] = 1
                right_mask[int(classes[random_idx[j + 1]]), random_idx[j]:random_idx[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])

    # Does the batch consists of more than a video?

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
        # print("next batch", self.index)
        batch_input = []
        batch_target = []
        batch_confidence = []
        video_names = []
        for vid in batch:
            video_name = vid.replace(".txt", "")
            video_names.append(video_name)  # Store video name
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch_confidence

    def get_video_names(self, batch_size):
        batch = self.list_of_examples[self.index - batch_size:self.index]
        video_names = []
        for vid in batch:
            video_name = vid.replace(".txt", "")
            video_names.append(video_name)  # Store video name
        # print(video_names)
        return video_names

    def get_single_random(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, vid in enumerate(batch):
            single_frame = self.random_index[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor

    def are_similar(self, batch_size, x, y):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
        video_names = []
        # print("are_similar", self.index)
        for vid in batch:
            video_names.append(vid)
        vid1_name = x
        vid2_name = y

        vid1_category = vid1_name.split("_")[-2]
        vid2_category = vid2_name.split("_")[-2]

        # Check if the categories are the same
        return vid1_category == vid2_category

    def get_boundary(self, batch_size, pred, epoch_num, dist_method='euclidean'):
        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, vid1 in enumerate(batch):
            # single_idx = self.random_index[vid1]
            vid1_gt = self.gt[vid1]
            features1 = pred[b].cpu().numpy()
            features1 = np.transpose(features1)
            features1 = np.squeeze(features1)
            features1 = features1[:len(vid1_gt)]
            boundary_target = np.ones(vid1_gt.shape) * (-100)

            similar_video_is_in_batch = False
            current_index = self.index
            for c, vid2 in enumerate(batch):
                if (vid1 != vid2) and self.are_similar(batch_size, vid1, vid2):
                    single_idx2 = self.random_index[vid2]
                    vid2_gt = self.gt[vid2]
                    features2 = pred[c].cpu().numpy()
                    features2 = np.transpose(features2)
                    features2 = np.squeeze(features2)
                    similar_video_is_in_batch = True
                    break
            self.index = current_index

            if (similar_video_is_in_batch == False):
                for d, vid2 in enumerate(self.list_of_examples):
                    if (vid1 != vid2) and self.are_similar(batch_size, vid1, vid2):
                        vid2_gt = self.gt[vid2]
                        vid2 = vid2.replace(".txt", "")
                        file_path = f"/content/drive/MyDrive/middle_out_results/middle_pred_{vid2}.npy"
                        features2 = np.load(file_path)
                        features2 = np.transpose(features2)
                        features2 = np.squeeze(features2)
                        break
                self.index = current_index

            n, m = features1.shape[0], features2.shape[0]
            dtw = np.zeros((n, m))
            dist = distance.cdist(features1, features2, metric=dist_method)

            # initialize the DTW matrix
            dtw[0, 0] = dist[0, 0]
            for i in range(1, n):
                dtw[i, 0] = dtw[i - 1, 0] + dist[i, 0]
            for j in range(1, m):
                dtw[0, j] = dtw[0, j - 1] + dist[0, j]

            # fill in the rest of the DTW matrix
            for i in range(1, n):
                for j in range(1, m):
                    cost = dist[i, j]
                    dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

            # backtrack to find the optimal warping path
            path = [(n - 1, m - 1)]
            i, j = n - 1, m - 1
            while i > 0 or j > 0:
                if i == 0:
                    cell = (i, j - 1)
                elif j == 0:
                    cell = (i - 1, j)
                else:
                    candidates = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
                    costs = [dtw[c[0], c[1]] for c in candidates]
                    argmin = np.argmin(costs)
                    cell = candidates[argmin]
                path.append(cell)
                i, j = cell

            # return the DTW distance, path, and matrix
            # dtw_distance = dtw[-1, -1]
            alignment_pairs = path[::-1]
            # dtw_matrix = dtw

            mapping_file = "/content/drive/MyDrive/data/gtea/mapping.txt"

            # MapLabelNumber = {}
            MapNumberLabel = {}
            with open(mapping_file, 'r') as mfp:
                for line in mfp:
                    values = line.strip().split()
                    if len(values) == 2:
                        num, label = values
                        try:
                            num = int(num)
                        except ValueError:
                            print(f"Invalid number '{num}' on line '{line}' from map file")
                            continue
                        # MapLabelNumber[label] = num
                        MapNumberLabel[num] = label

            colors = cc.glasbey_light[:len(MapNumberLabel)]

            # for vid1 in similar_videos:
            # video1, video2 = vid1, vid2
            # video1_content = []
            # video2_content = []
            #
            # with open(f"/content/drive/MyDrive/data/gtea/groundTruth/{video1}", 'r') as lfp:
            #     for line in lfp:
            #         if line.startswith("#"):
            #             continue
            #         for word in line.split():
            #             video1_content.append(MapLabelNumber[word])
            #
            if not vid2.endswith(".txt"):
                vid2 += ".txt"
            # with open(f"/content/drive/MyDrive/data/gtea/groundTruth/{video2}", 'r') as lfp:
            #     for line in lfp:
            #         if line.startswith("#"):
            #             continue
            #         for word in line.split():
            #             video2_content.append(MapLabelNumber[word])

            annotation_file_path = r"/content/TimestampActionSeg/data/gtea_annotation_all.npy"
            annotations = np.load(annotation_file_path, allow_pickle=True).item()

            timestamp_frames_video1 = []
            timestamp_frames_video2 = []

            for video, annotation in annotations.items():
                if video == vid1:
                    for timestamp_frame in enumerate(annotation):
                        timestamp_frames_video1.append((timestamp_frame[1], vid1_gt[timestamp_frame[1]]))

            for video, annotation in annotations.items():
                if video == vid2:
                    for timestamp_frame in enumerate(annotation):
                        timestamp_frames_video2.append(timestamp_frame[1])

            timestamp_pairs = []  # can be deleted; for visualisation only
            generated_timestamps = []
            for i in timestamp_frames_video2:

                for tuple in alignment_pairs[i:]:  # (frame from video1, frame from video2) = tuple
                    if tuple[1] == i:
                        timestamp_pairs.append(tuple)
                        generated_timestamps.append((tuple[0], vid2_gt[tuple[1]]))  # (frame video1, action video2)
                        break
            # generated timestamp frame =100
            # original timestamp frame = 100

            new_timestamps = []  # needed timestamps from generated timestamps
            for tuple in generated_timestamps:  # frame from video1, action label from video2
                index_dtw = 0
                while index_dtw < (len(timestamp_frames_video1) - 1) and tuple[0] > \
                        timestamp_frames_video1[index_dtw][0]:
                    index_dtw += 1
                if index_dtw == 0:
                    if (tuple[1] == timestamp_frames_video1[index_dtw][1]):
                        try:
                            if tuple[0] == timestamp_frames_video1[index_dtw][0]:
                                new_timestamps.append((tuple[0] + 1, tuple[1]))
                            else:
                                new_timestamps.append(tuple)
                        except IndexError:
                            print("index error")
                            print("///////////////////****************************/////////////////////**************")
                            break
                else:
                    if (tuple[0] == timestamp_frames_video1[index_dtw][0]):
                        if tuple[1] == timestamp_frames_video1[index_dtw - 1][1]:
                            new_timestamps.append((tuple[0] - 1, tuple[1]))
                        elif tuple[1] == timestamp_frames_video1[index_dtw + 1][1]:
                            new_timestamps.append((tuple[0] + 1, tuple[1]))
                    else:
                        if (tuple[1] == timestamp_frames_video1[index_dtw][1]) or (
                                tuple[1] == timestamp_frames_video1[index_dtw - 1][1]):
                            new_timestamps.append((tuple[0], tuple[1]))

            # if (vid1 == "S4_Coffee_C1.txt"):
            #     print(new_timestamps)
            #     print(generated_timestamps)

            for index, timestamp in enumerate(new_timestamps):
                if index == 0:
                    continue
                # print(index)
                try:
                    index_dtw = 0
                    while index_dtw < (len(timestamp_frames_video1) - 1) and timestamp[0] > \
                            timestamp_frames_video1[index_dtw][0]:
                        index_dtw += 1
                    if (timestamp[1] != timestamp_frames_video1[index_dtw][1]) and (
                            new_timestamps[index - 1][0] >= timestamp_frames_video1[index_dtw - 1][0]) \
                            and (new_timestamps[index - 1][1] != timestamp_frames_video1[index_dtw - 1][1]) and (
                            timestamp[0] <= timestamp_frames_video1[index_dtw][0]) \
                            and (timestamp[1] == timestamp_frames_video1[index_dtw - 1][1]) and (
                            new_timestamps[index - 1][1] == timestamp_frames_video1[index_dtw][1]):
                        # print(vid1, index_dtw, index)
                        # print(timestamp_frames_video1[index_dtw - 1][1], new_timestamps[index - 1][1],
                        # new_timestamps[index][1], timestamp_frames_video1[index_dtw][1])
                        new_tuple1 = (new_timestamps[index][0] + 1, new_timestamps[index - 1][1])
                        new_tuple2 = (new_timestamps[index - 1][0] - 1, new_timestamps[index][1])
                        del new_timestamps[index]
                        del new_timestamps[index - 1]
                        new_timestamps.insert(index - 1, new_tuple2)
                        new_timestamps.insert(index, new_tuple1)
                        new_timestamps = sorted(new_timestamps, key=lambda x: x[0])

                except IndexError:
                    print("index error in solution 1")
                    continue

            # if (vid1 == "S4_Coffee_C1.txt"):
            #     print(new_timestamps)
            #     print(generated_timestamps)

            # new_timestamps = sorted(new_timestamps, key=lambda x: x[0])

            for tuple in new_timestamps:
                index_dtw = 0
                while index_dtw < (len(timestamp_frames_video1) - 1) and tuple[0] > \
                        timestamp_frames_video1[index_dtw][0]:
                    index_dtw += 1
                if (tuple[0] != timestamp_frames_video1[index_dtw][0]):
                    timestamp_frames_video1.append(tuple)

            timestamp_frames_video1 = sorted(timestamp_frames_video1, key=lambda x: x[0])

            for i in range(0, timestamp_frames_video1[0][0]):
                boundary_target[i] = timestamp_frames_video1[0][1]  # assign labels to frames from 0 to first timestamp

            for i in range(0, len(timestamp_frames_video1) - 1):
                if timestamp_frames_video1[i][1] == timestamp_frames_video1[i + 1][1]:
                    for j in range(timestamp_frames_video1[i][0] + 1, timestamp_frames_video1[i + 1][0]):
                        boundary_target[j] = timestamp_frames_video1[i][
                            1]  # assigning labels to frames between two similar timestamps
                boundary_target[timestamp_frames_video1[i][0]] = timestamp_frames_video1[i][
                    1]  # assign labels to frames with timestamp

            if (timestamp_frames_video1[-1][1] == timestamp_frames_video1[-2][
                1]):  # assign labels to frames from 0 to first timestamp
                for i in range(timestamp_frames_video1[-2][0], len(vid1_gt)):
                    boundary_target[i] = timestamp_frames_video1[-1][1]
            else:
                boundary_target[timestamp_frames_video1[-2][0]] = timestamp_frames_video1[-2][
                    1]  # in case last two timestamps are bot equal,
                # assign label to timestamp before last
                for i in range(timestamp_frames_video1[-1][0], len(vid1_gt)):
                    boundary_target[i] = timestamp_frames_video1[-1][1]

            boundary_target_tensor[b, :vid1_gt.shape[0]] = torch.from_numpy(boundary_target)
            if (epoch_num % 10 == 0):
                print(vid1)
                # print(timestamp_frames_video2)
                MaxFrames = max(len(vid1_gt), len(vid2_gt))

                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 2))

                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

                # plotting the first subplot
                x_pos = 0
                for label in range(0, len(vid1_gt)):
                    if boundary_target[label] == -100:
                        ax1.barh(0, 1, left=x_pos, height=1, color='#FFFFFF')
                    else:
                        ax1.barh(0, 1, left=x_pos, height=1, color=colors[int(boundary_target[label])])
                    x_pos += 1

                ax1.set_xlim([0, MaxFrames])
                ax1.set_yticks([])
                ax1.set_xticks([])

                x_pos = 0
                for label in vid2_gt:
                    ax2.barh(0, 1, left=x_pos, height=1, color=colors[int(label)])
                    x_pos += 1
                ax2.set_xlim([0, MaxFrames])
                ax2.set_yticks([])
                ax2.set_xticks([])

                x_pos = 0
                for label in vid1_gt:
                    ax3.barh(0, 1, left=x_pos, height=1, color=colors[int(label)])
                    x_pos += 1
                ax3.set_xlim([0, MaxFrames])
                ax3.set_yticks([])
                ax3.set_xticks([])

                for tuple in timestamp_pairs:
                    xyA = (tuple[0] / MaxFrames, -0.5)  # Center of the first subplot
                    xyB = (tuple[1] / MaxFrames, 0.5)  # Left edge of the second subplot

                    # Create the ConnectionPatch object with the arrow properties
                    con = ConnectionPatch(
                        xyA=xyA, coordsA=ax1.get_yaxis_transform(),
                        xyB=xyB, coordsB=ax2.get_yaxis_transform(),
                        arrowstyle="<|-|>,head_width=0.2", color='black')

                    # # Add the arrow to the second subplot
                    ax2.add_artist(con)

                x_pos = 0.02
                for i in MapNumberLabel:
                    ax3.text(x_pos, -0.24, MapNumberLabel[i], ha='center', color=colors[i], transform=ax3.transAxes)
                    x_pos += (len(MapNumberLabel[i]) + 1) / 70

                # delete the boarders
                ax1.spines["top"].set_visible(False)
                ax1.spines["bottom"].set_visible(False)
                ax1.spines["left"].set_visible(False)
                ax1.spines["right"].set_visible(False)

                ax2.spines["top"].set_visible(False)
                ax2.spines["bottom"].set_visible(False)
                ax2.spines["left"].set_visible(False)
                ax2.spines["right"].set_visible(False)

                ax3.spines["top"].set_visible(False)
                ax3.spines["bottom"].set_visible(False)
                ax3.spines["left"].set_visible(False)
                ax3.spines["right"].set_visible(False)

                plt.savefig(
                    "/content/drive/MyDrive/visualise/" + vid1.split('.txt')[0] + "_epoch" + str(epoch_num) + ".png")

                # for i in range(0,len(video1_content))

        return boundary_target_tensor

        # return dtw_content



