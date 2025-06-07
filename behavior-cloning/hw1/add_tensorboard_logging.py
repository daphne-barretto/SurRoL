import sys
import os

# Get the path to the training script
script_path = sys.argv[1]

# Read the current content
with open(script_path, 'r') as f:
    content = f.read()

# Add TensorBoard import
if 'import torch.utils.tensorboard' not in content:
    tensorboard_import = "import torch.utils.tensorboard as tensorboard"
    content = content.replace("import time", "import time\n" + tensorboard_import)

# Update the train_bc_agent function to include TensorBoard writer
tensorboard_writer_code = """
    # Create TensorBoard writer
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(save_path), 'logs')
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs will be saved to {tensorboard_dir}")
"""

# Insert the code after the logger creation
content = content.replace("logger = Logger(log_dir)", "logger = Logger(log_dir)\n" + tensorboard_writer_code)

# Add TensorBoard logging for metrics
tensorboard_logging_code = """
        # Log to TensorBoard
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('validation/loss', avg_val_loss, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
"""

# Insert the code after the logger.log_scalar calls
content = content.replace("logger.log_scalar(current_lr, \"learning_rate\", epoch)", 
                         "logger.log_scalar(current_lr, \"learning_rate\", epoch)\n" + tensorboard_logging_code)

# Add model graph to TensorBoard
model_graph_code = """
    # Add model graph to TensorBoard
    dummy_obs = policy.process_observation(all_obs[0:2])
    dummy_obs_tensor = ptu.from_numpy(dummy_obs.astype(np.float32))
    writer.add_graph(policy, dummy_obs_tensor)
"""

# Insert the code after the model creation
content = content.replace("print(f\"Goal dimension: {goal_dim}\")", "print(f\"Goal dimension: {goal_dim}\")\n" + model_graph_code)

# Add closing the writer
writer_close_code = """
    # Close TensorBoard writer
    writer.close()
"""

# Insert the code before returning the policy
content = content.replace("return policy", writer_close_code + "\n    return policy")

# Write the modified content back to the file
with open(script_path, 'w') as f:
    f.write(content)

print(f"Added TensorBoard logging to {script_path}")
