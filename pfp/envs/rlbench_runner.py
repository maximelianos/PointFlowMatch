import wandb
from tqdm import tqdm
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.policy.base_policy import BasePolicy

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())


class RLBenchRunner:
    def __init__(
        self,
        num_episodes: int,
        max_episode_length: int,
        env_config: dict,
        verbose=False,
    ) -> None:
        self.env: RLBenchEnv = RLBenchEnv(**env_config)
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.verbose = verbose

        self.use_droid = True

        # MV
        if self.use_droid:
            self.is_train: bool = False
            from droidloader.train_loader import EpisodeList
            self.episodes = EpisodeList(self.is_train)
            print(">>> create dataloader, is_train", self.is_train, "len", len(self.episodes))
            input()

        return

    def run(self, policy: BasePolicy):
        wandb.define_metric("success", summary="mean")
        wandb.define_metric("steps", summary="mean")
        success_list: list[bool] = []
        steps_list: list[int] = []
        self.env.reset_rng()

        import numpy as np
        from droidloader.eval_logger import EvalLogger, eval_results

        eval_logger = EvalLogger()

        for idx in range(len(self.episodes)):
            sample = self.episodes[idx]
            eval_logger.vis_start(eval_results.episode_idx)

            obs = sample["robot_state"] # (n_steps, robot_state)
            pcd = sample["pcd_xyz"] # (n_steps, n_points, 3)
            pcd = pcd[0, :4096]

            result_traj = np.zeros_like(obs)
            result_traj[0] = obs[0]

            step = 1
            while step < len(obs):
                prediction = policy.predict_action(pcd, result_traj[step-1])  # (K, n_pred_steps, robot_state)
                _pred_len = prediction.shape[1]
                for i in range(_pred_len):
                    if step < len(obs):
                        result_traj[step] = prediction[-1, i]
                        step += 1
                    else:
                        break
                break
            # robot_state = obs[0]
            # #eval_logger.vis_step(robot_state)
            # for step in range(1, len(obs)):
            #     prediction = policy.predict_action(pcd, robot_state)  # (K, n_pred_steps, robot_state)
            #     _pred_len = len(obs)
            #     result_traj[1:_pred_len] = prediction[-1, :_pred_len-1]
            #     break
            #     robot_state = prediction[-1, 0]
            #     result_traj[step] = robot_state
            #     eval_logger.vis_step(robot_state)
            #
            for step in range(len(obs)):
                eval_logger.vis_step(result_traj[step])
            imginfo(obs)
            imginfo(result_traj)
            eval_results.pred = result_traj
            eval_logger.vis_stop()

        return

        for episode in tqdm(range(self.num_episodes)):
            print(episode)
            input()
            policy.reset_obs()
            self.env.reset()
            for step in range(self.max_episode_length):
                print("step", step)
                input()
                robot_state, obs = self.env.get_obs()
                prediction = policy.predict_action(obs, robot_state)
                imginfo(robot_state)
                imginfo(prediction)
                # self.env.vis_step(robot_state, obs, prediction)
                # next_robot_state = prediction[-1, 0]  # Last K step, first T step
                # reward, terminate = self.env.step(next_robot_state)
                # success = bool(reward)
                # if success or terminate:
                #     break
            # success_list.append(success)
            # if success:
            #     steps_list.append(step)
            # if self.verbose:
            #     print(f"Steps: {step}")
            #     print(f"Success: {success}")
            # wandb.log({"episode": episode, "success": int(success), "steps": step})
        return success_list, steps_list
