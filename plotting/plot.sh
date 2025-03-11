# set logs directory
LOGS_DIR="C:\Users\dbarretto\SurRoL\logs"

python plotting/plot_task.py "$LOGS_DIR" NeedleReach
python plotting/plot_task.py "$LOGS_DIR" NeedleReach --plot_separate
python plotting/plot_task.py "$LOGS_DIR" GauzeRetrieve
python plotting/plot_task.py "$LOGS_DIR" GauzeRetrieve --plot_separate
python plotting/plot_task.py "$LOGS_DIR" NeedlePick
python plotting/plot_task.py "$LOGS_DIR" NeedlePick --plot_separate
python plotting/plot_task.py "$LOGS_DIR" PegTransfer
python plotting/plot_task.py "$LOGS_DIR" PegTransfer --plot_separate