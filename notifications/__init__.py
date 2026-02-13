"""
Notifications Package
=====================

Provides notification services for trading alerts and reports.

Available Notifiers:
    - EmailNotifier: Send email notifications via SMTP
"""

from .email_notifier import (
    EmailNotifier,
    RebalanceAction,
    RebalanceReport,
    create_email_notifier,
)

__all__ = [
    'EmailNotifier',
    'RebalanceAction',
    'RebalanceReport',
    'create_email_notifier',
]
