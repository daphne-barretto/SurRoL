"""
Implementation of the missing functions in bc_trainer.py
"""

def collect_training_trajectories(
        self,
        itr,
        load_initial_expertdata,
        collect_policy
):
    """
    :param itr:
    :param load_initial_expertdata: path to expert data pkl file
    :param collect_policy: the current policy using which we collect data
    :return:
        paths: a list trajectories
        envsteps_this_batch: the sum over the numbers of environment steps in paths
        train_video_paths: paths which also contain videos for visualization purposes
    """

    # On the first iteration, load expert data
    if itr == 0 and load_initial_expertdata is not None:
        paths = pickle.load(open(load_initial_expertdata, 'rb'))
        envsteps_this_batch = 0
        for path in paths:
            envsteps_this_batch += len(path['reward'])
        return paths, envsteps_this_batch, None
    
    # On future iterations (during DAgger), collect more data
    print("\nCollecting data to be used for training...")
    paths, envsteps_this_batch = utils.sample_trajectories(
        self.env, collect_policy, self.params['batch_size'], self.params['ep_len'])

    # collect more rollouts with the same policy, to be saved as videos in tensorboard
    # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
    train_video_paths = None
    if self.log_video:            
        print('\nCollecting train rollouts to be used for saving videos...')
        train_video_paths = utils.sample_n_trajectories(self.env,
            collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

    return paths, envsteps_this_batch, train_video_paths

def train_agent(self):
    """
    Samples a batch of trajectories and updates the agent with the batch
    """
    print('\nTraining agent using sampled data from replay buffer...')
    all_logs = []
    for train_step in range(self.params['num_agent_train_steps_per_iter']):

        # Sample some data from the data buffer
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
            self.params['train_batch_size'])

        # Use the sampled data to train the agent
        train_log = self.agent.train(ob_batch, ac_batch)
        all_logs.append(train_log)
    
    return all_logs

def do_relabel_with_expert(self, expert_policy, paths):
    """
    Relabels collected trajectories with an expert policy

    :param expert_policy: the policy we want to relabel the paths with
    :param paths: paths to relabel
    """
    expert_policy.to(ptu.device)
    print("\nRelabelling collected observations with labels from an expert policy...")

    # Relabel collected observations with labels from an expert policy
    for i in range(len(paths)):
        observations = paths[i]["observation"]
        expert_actions = expert_policy.get_action(observations)
        paths[i]["action"] = expert_actions
    
    return paths