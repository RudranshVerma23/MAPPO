import params
import torch

class CurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_stage = 1
        self.plateau_counter = 0
        self.best_reward = -float('inf')
        
        # Stage Thresholds
        self.patience = 20  # Number of batches to wait for plateau
        self.qos_threshold = 0.90  # 90% QoS satisfaction needed to advance

    def update_stage(self, avg_reward, qos_satisfaction):
        """
        Logic to advance the curriculum based on performance.
        """
        # Check for plateau
        if avg_reward > self.best_reward + 0.01:
            self.best_reward = avg_reward
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        # current_target = 0.70 if self.current_stage == 1 else self.qos_threshold
        current_target = 0.90
        
        # Transition Trigger: Plateau reached AND QoS satisfied
        if self.plateau_counter >= self.patience and qos_satisfaction >= current_target:
            if self.current_stage < 4:
                self.current_stage += 1
                self._apply_stage_logic()
                self.plateau_counter = 0
                self.best_reward = -float('inf') # Reset for new stage
                print(f"--- CURRICULUM ADVANCED TO STAGE {self.current_stage} ---")

    def _apply_stage_logic(self):
        """
        Directly modifies environment attributes for the next stage.
        """
        self.env.current_stage = self.current_stage
        if self.current_stage == 2:
            # Unlock 3D and Power Allocation
            # (Handled by removing action masking in the step function)
            print("Action Space Unlocked: 3D Flight & NOMA Power Allocation active.")
            pass 
        elif self.current_stage == 3:
            # ISAC Integration: Enable Sensing by adjusting Alpha
            # Weight shifts from 1.0 (Comm-only) to 0.5 (Balanced) [cite: 144]
            self.env.alpha = 0.5 
            print("Objective Shift: Sensing (MI) now contributes to reward.")
        elif self.current_stage == 4:
            # Dynamic Survivor: Unlock mobility 
            self.env.mobility_factor = 0.1 # Start annealing from 0 to 1