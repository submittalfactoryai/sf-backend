# background_tasks.py
"""
Background scheduler for checking and locking expired trial accounts.
Runs once per night at 02:00 UTC.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from database import SessionLocal
from services.subscription_service import SubscriptionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("background_tasks")


# ----------------------------------------------------------------------
# TASK: Check expired trials
# ----------------------------------------------------------------------
def check_expired_trials():
    """Check and lock all expired trial accounts."""
    db = SessionLocal()
    try:
        locked_count = SubscriptionService.check_expired_trials_bulk(db)
        logger.info(f"[Scheduler] Trial expiry check completed: {locked_count} users locked")
    except Exception as e:
        logger.error(f"[Scheduler] Error checking expired trials: {e}")
    finally:
        db.close()


# ----------------------------------------------------------------------
# SCHEDULER STARTUP — RUNS NIGHTLY AT 02:00 UTC
# ----------------------------------------------------------------------
def start_background_scheduler():
    """
    Start background scheduler for periodic tasks.
    Runs every night at 02:00 UTC.
    Call inside your FastAPI main startup.
    """
    try:
        scheduler = BackgroundScheduler()

        scheduler.add_job(
            check_expired_trials,
            trigger=CronTrigger(hour=2, minute=0),  # 🌙 Every night at 02:00 UTC
            id="check_expired_trials",
            name="Nightly trial-expiry check",
            replace_existing=True,
        )

        scheduler.start()
        logger.info("[Scheduler] Started — will run nightly at 02:00 UTC")

        return scheduler

    except Exception as e:
        logger.error(f"[Scheduler] Failed to start: {e}")
        return None


# ----------------------------------------------------------------------
# Manual run for testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    check_expired_trials()
