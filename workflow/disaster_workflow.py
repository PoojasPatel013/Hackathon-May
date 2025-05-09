import schedule
import time
import logging
import traceback
import requests
from datetime import datetime
from pythonjsonlogger import jsonlogger

# Import your project modules
from predict import DisasterRiskPredictor

# Advanced JSON Logging
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # JSON log handler
    json_handler = logging.FileHandler('disaster_workflow.json')
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(message)s %(filename)s %(lineno)d'
    )
    json_handler.setFormatter(formatter)
    logger.addHandler(json_handler)

    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger

# Configure logging
logger = setup_logger()

class WorkflowError(Exception):
    """Custom exception for workflow-related errors"""
    pass

class DisasterWorkflow:
    def __init__(self, max_retries=3, notification_webhook=None):
        """
        Initialize workflow with error handling and notification capabilities
        
        Args:
            max_retries (int): Maximum number of retry attempts
            notification_webhook (str): Webhook URL for error notifications
        """
        self.predictor = DisasterRiskPredictor()
        self.last_run = None
        self.max_retries = max_retries
        self.retry_count = 0
        self.notification_webhook = notification_webhook

    def send_error_notification(self, error_message):
        """
        Send error notification via webhook
        
        Args:
            error_message (str): Detailed error description
        """
        if not self.notification_webhook:
            logger.warning("No notification webhook configured")
            return

        try:
            payload = {
                "text": f"""🚨 *Disaster Prediction Workflow Error* 🚨
                
*Timestamp:* {datetime.now()}
*Error Details:*
```
{error_message}
```

*Retry Count:* {self.retry_count}
""",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🚨 *Workflow Error Detected* (Retry {self.retry_count})"
                        }
                    }
                ]
            }

            response = requests.post(
                self.notification_webhook, 
                json=payload,
                timeout=10
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to send notification: {response.text}")
        
        except Exception as notification_error:
            logger.error(f"Notification sending failed: {notification_error}")

    def run_daily_workflow(self):
        """
        Execute daily disaster risk prediction workflow
        with robust error handling and retry mechanism
        """
        try:
            logger.info("Starting daily disaster risk prediction workflow")
            
            # Data Collection
            data = self.predictor.collect_data()
            logger.info(f"Collected {len(data)} data points")
            
            # Model Training (if needed)
            self.predictor.train_model()
            logger.info("Model training completed")
            
            # Risk Prediction
            risk_predictions = self.predictor.predict_global_risks()
            logger.info(f"Generated {len(risk_predictions)} risk predictions")
            
            # Save Predictions
            self.predictor.save_predictions(risk_predictions)
            
            # Reset retry count on successful run
            self.retry_count = 0
            self.last_run = datetime.now()
            logger.info("Daily workflow completed successfully")
        
        except Exception as e:
            error_msg = f"Workflow Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Retry mechanism
            self.retry_count += 1
            if self.retry_count <= self.max_retries:
                logger.warning(f"Retrying workflow. Attempt {self.retry_count}")
                time.sleep(60)  # Wait 1 minute before retry
                self.run_daily_workflow()
            else:
                # Send critical error notification
                self.send_error_notification(error_msg)
                self.retry_count = 0

    def run_monthly_retrain(self):
        """
        Execute monthly model retraining
        with robust error handling
        """
        try:
            logger.info("Starting monthly model retraining")
            
            # Full model retraining
            self.predictor.collect_data(full_dataset=True)
            self.predictor.train_model(retrain=True)
            
            logger.info("Monthly model retraining completed")
        
        except Exception as e:
            error_msg = f"Monthly Retraining Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.send_error_notification(error_msg)

def main():
    # Optional: Configure Slack/Discord webhook for notifications
    notification_webhook = "https://your-webhook-url.com"
    
    workflow = DisasterWorkflow(
        max_retries=3, 
        notification_webhook=notification_webhook
    )
    
    # Daily workflow at 2 AM
    schedule.every().day.at("02:00").do(workflow.run_daily_workflow)
    
    # Monthly retraining on first day of month at 3 AM
    schedule.every().month.at("03:00").do(workflow.run_monthly_retrain)
    
    logger.info("Disaster prediction workflow scheduler started")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()