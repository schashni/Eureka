class Cartpole(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()
        self.object_pos = self.obs_buf.clone()

        # Goal state (stationary cart at center, upright and stationary pole)
        goal_cart_pos = 0
        goal_cart_vel = 0
        goal_pole_angle = np.pi / 2  # Upright pole
        goal_pole_ang_vel = 0
        self.goal_pos = torch.tensor([[goal_cart_pos, goal_cart_vel, goal_pole_angle, goal_pole_ang_vel]] * self.num_envs, device=self.device)

        return self.obs_buf
