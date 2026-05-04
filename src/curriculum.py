import torch
import params


class CurriculumManager:
    """
    T8 fixes:
    - patience 20->10
    - Stage-1 QoS threshold 0.70->0.50
    - Advance on plateau OR QoS met (after min_stage_epochs)
    - Logging of plateau counter each call
    """

    def __init__(self, env, min_stage_epochs=30):
        self.env = env
        self.current_stage = getattr(env, 'current_stage', 1)
        self.plateau_counter = 0
        self.best_reward = -float('inf')
        self.patience = 10
        self.qos_threshold = 0.75
        self.stage1_qos_target = 0.50
        self.min_stage_epochs = min_stage_epochs
        self._stage_epoch = 0

    def update_stage(self, avg_reward, qos_satisfaction):
        self._stage_epoch += 1

        if avg_reward > self.best_reward + 1e-6:
            self.best_reward = avg_reward
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        current_target = (
            self.stage1_qos_target if self.current_stage == 1
            else self.qos_threshold
        )

        plateau_ready = self.plateau_counter >= self.patience
        qos_ready     = qos_satisfaction >= current_target
        min_done      = self._stage_epoch >= self.min_stage_epochs

        if min_done and (plateau_ready or qos_ready):
            if self.current_stage < 4:
                self.current_stage += 1
                self._apply_stage_logic()
                self.plateau_counter = 0
                self.best_reward = -float('inf')
                self._stage_epoch = 0
                print(f"--- CURRICULUM -> STAGE {self.current_stage} "
                      f"(plateau={plateau_ready}, qos={qos_satisfaction*100:.1f}%) ---")
            else:
                if plateau_ready:
                    self.env.mobility_factor = min(
                        getattr(self.env, 'mobility_factor', 0.0) + 0.1, 1.0
                    )
                    self.plateau_counter = 0
                    self.best_reward = -float('inf')
                    print(f"--- STAGE 4: mobility_factor -> {self.env.mobility_factor:.2f} ---")

    def _apply_stage_logic(self):
        self.env.current_stage = self.current_stage
        if self.current_stage == 2:
            print("  Stage 2: 3D flight + NOMA power control unlocked.")
        elif self.current_stage == 3:
            self.env.alpha = 0.5
            print("  Stage 3: alpha=0.5 active.")
        elif self.current_stage == 4:
            self.env.mobility_factor = min(
                getattr(self.env, 'mobility_factor', 0.0) + 0.1, 1.0
            )
            print(f"  Stage 4: mobility_factor={self.env.mobility_factor:.2f}.")