def save_episodes_hdf5(all_data, camera_names, save_dir="collected_data_hdf5", rate=30):
    """
    Saves each episode (traj, camera_data) to an HDF5 file with consistent shapes,
    padding all episodes to the same trajectory length with zeros.

    1) We do two passes. First pass finds the maximum T-1 among all episodes.
    2) Second pass: For each episode, create a dataset of shape (max_timesteps, ...)
       and fill from [0..(T-1)], then pad the rest with zeros.

    :param all_data: list of episodes, where each episode is (traj, camera_data)
                     - traj: shape (T, 7) or similar
                     - camera_data: length T list, each dict {cam_name -> (3,H,W)}
    :param camera_names: list of camera name strings (like ["cam0"] or ["cam0", "cam1"]).
    :param save_dir: folder to save the .hdf5 files
    :param rate: capture rate (Hz) to compute qvel if needed
    """
    import os
    import h5py
    os.makedirs(save_dir, exist_ok=True)
    dt = 1.0 / rate

    # --------------------------
    # PASS 1: find max_timesteps
    # --------------------------
    max_len = 0  # store the largest (T - 1) found
    for (traj, _) in all_data:
        T = len(traj)
        if T < 2:
            continue
        # The number of "steps" we store is T-1
        length = T - 1
        if length > max_len:
            max_len = length

    print(f"Global max trajectory length (T-1) across all episodes = {max_len}")
    if max_len == 0:
        print("No valid episodes. Nothing to save.")
        return

    # ---------------------------
    # PASS 2: create & pad
    # ---------------------------
    for episode_idx, (traj, camera_data) in enumerate(all_data):
        T = len(traj)
        if T < 2:
            print(f"Skipping episode {episode_idx}, not enough steps.")
            continue

        this_len = T - 1  # number of time steps to store
        # We'll create arrays with shape (max_len, …) and fill [0..this_len), then pad the rest

        # Prepare our data arrays, all with shape (max_len, …)
        # qpos, qvel, action => shape (max_len, 7) [assuming 7-dim robot states]
        # images => shape (max_len, H, W, 3)
        # We'll fill them with zeros, then copy the actual data in the front

        # We'll build them as python lists, then convert to np.array
        # or build them as np.zeros(...) directly.

        qpos_list   = np.zeros((max_len, 8), dtype=np.float32)
        qvel_list   = np.zeros((max_len, 8), dtype=np.float32)
        action_list = np.zeros((max_len, 8), dtype=np.float32)

        # For images, we need to know H,W. We'll get it from the first frame
        # of the first camera, after we parse the data. But let's do it after we read them.

        # We'll store the images in python lists, then pad them after we know their shape.
        images_dict = {
            cam: [] for cam in camera_names
        }

        # Fill the first [this_len] from the data
        for t in range(this_len):
            qpos_t = traj[t]
            qpos_next = traj[t + 1]
            qpos_list[t] = qpos_t

            # qvel[t] = (qpos[t+1] - qpos[t]) / dt
            qvel_list[t] = (qpos_next - qpos_t) / dt

            action_list[t] = qpos_next

            # images
            frame_dict = camera_data[t]  # {cam_name: (3,H,W)}
            for cam in camera_names:
                frame_chw = frame_dict[cam]
                frame_hwc = frame_chw.transpose(1, 2, 0)  # (H,W,3)
                if frame_hwc.dtype != np.uint8:
                    frame_hwc = frame_hwc.astype(np.uint8)
                images_dict[cam].append(frame_hwc)

        # Now let's see how many images we have in each camera: this_len
        # We must pad with zero images up to max_len.
        # We'll do that by checking the shape of the first real image,
        # then making blank 0-coded images for the remainder.
        for cam in camera_names:
            if len(images_dict[cam]) == 0:
                print(f"Warning: Episode {episode_idx} has no images for cam '{cam}'.")
                continue

            # shape of the first real image
            first_img = images_dict[cam][0]
            H, W, C = first_img.shape
            # If we have fewer than max_len images, pad the rest
            needed_pad = max_len - this_len
            if needed_pad > 0:
                pad_img = np.zeros((H, W, C), dtype=np.uint8)
                for _ in range(needed_pad):
                    images_dict[cam].append(pad_img)

        # Convert images_dict[cam] to a single array shape (max_len, H, W, 3)
        # e.g. images_list => np.stack(...)
        final_images = {}
        for cam in camera_names:
            if len(images_dict[cam]) < max_len:
                # Something odd happened, fill with zeros
                needed_pad = max_len - len(images_dict[cam])
                if needed_pad > 0 and len(images_dict[cam]) > 0:
                    H, W, C = images_dict[cam][0].shape
                    pad_img = np.zeros((H, W, C), dtype=np.uint8)
                    for _ in range(needed_pad):
                        images_dict[cam].append(pad_img)
                else:
                    # no valid images at all
                    pass

            # shape => (max_len, H, W, 3)
            final_images[cam] = np.stack(images_dict[cam], axis=0)

        # Now we can create the HDF5 with shape = (max_len, …)
        file_path = os.path.join(save_dir, f"episode_{episode_idx}.hdf5")
        print(f"Saving padded episode {episode_idx} to {file_path}")

        with h5py.File(file_path, 'w') as root:
            root.attrs['sim'] = False  # or True, whichever

            obs_group = root.create_group('observations')
            img_group = obs_group.create_group('images')

            # For each cam, shape => (max_len, H, W, 3)
            for cam in camera_names:
                if cam in final_images:
                    arr = final_images[cam]
                    ds = img_group.create_dataset(
                        cam,
                        data=arr,  # directly set from array
                        shape=arr.shape,
                        dtype='uint8',
                        chunks=(1, arr.shape[1], arr.shape[2], arr.shape[3])
                    )

            qpos_ds = obs_group.create_dataset('qpos', data=qpos_list)
            qvel_ds = obs_group.create_dataset('qvel', data=qvel_list)
            act_ds  = root.create_dataset('action', data=action_list)

        print(f"Done saving episode {episode_idx}, length={this_len}, padded to {max_len}.\n")