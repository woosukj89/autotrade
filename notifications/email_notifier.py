"""
Email Notification Service
==========================

Sends email notifications for trading activity, rebalancing reports,
and alerts.

Configuration via environment variables:
    - SMTP_HOST: SMTP server host (default: smtp.gmail.com)
    - SMTP_PORT: SMTP server port (default: 587)
    - SMTP_USERNAME: SMTP username/email
    - SMTP_PASSWORD: SMTP password or app-specific password
    - SMTP_FROM_EMAIL: Sender email address (defaults to SMTP_USERNAME)
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RebalanceAction:
    """Represents a single rebalancing action."""
    ticker: str
    action: str  # "BUY" or "SELL"
    shares: float
    price: float
    value: float
    reason: str


@dataclass
class RebalanceReport:
    """Complete rebalancing report."""
    timestamp: datetime
    strategy_name: str

    # Regime information
    bear_score: float
    risk_level: str
    allocation_aggressive: float
    allocation_defensive: float

    # Actions taken
    actions: List[RebalanceAction]

    # Portfolio state
    portfolio_value: float
    cash: float
    positions: Dict[str, dict]  # ticker -> {shares, value, weight}

    # Reasons
    rebalance_reason: str
    regime_change: bool


class EmailNotifier:
    """
    Email notification service for trading alerts and reports.

    Usage:
        notifier = EmailNotifier(recipient_email="user@example.com")

        # Send a rebalance report
        notifier.send_rebalance_report(report)

        # Send a simple alert
        notifier.send_alert("Trade Executed", "Bought 10 shares of AAPL")
    """

    def __init__(
        self,
        recipient_email: str,
        smtp_host: str = None,
        smtp_port: int = None,
        smtp_username: str = None,
        smtp_password: str = None,
        from_email: str = None,
    ):
        """
        Initialize email notifier.

        Args:
            recipient_email: Email address to send notifications to
            smtp_host: SMTP server host (default from env or smtp.gmail.com)
            smtp_port: SMTP server port (default from env or 587)
            smtp_username: SMTP username (default from env)
            smtp_password: SMTP password (default from env)
            from_email: Sender email address (default: smtp_username)
        """
        self.recipient_email = recipient_email
        self.smtp_host = smtp_host or os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.environ.get('SMTP_PORT', '587'))
        self.smtp_username = smtp_username or os.environ.get('SMTP_USERNAME')
        self.smtp_password = smtp_password or os.environ.get('SMTP_PASSWORD')
        self.from_email = from_email or os.environ.get('SMTP_FROM_EMAIL', self.smtp_username)

        if not self.smtp_username or not self.smtp_password:
            print("[EmailNotifier] Warning: SMTP credentials not configured.")
            print("  Set SMTP_USERNAME and SMTP_PASSWORD environment variables.")
            print("  For Gmail, use an App Password: https://support.google.com/accounts/answer/185833")

    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.smtp_username and self.smtp_password and self.recipient_email)

    def send_email(self, subject: str, body_html: str, body_text: str = None) -> bool:
        """
        Send an email.

        Args:
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body (optional, derived from HTML if not provided)

        Returns:
            True if email sent successfully, False otherwise.
        """
        if not self.is_configured():
            print("[EmailNotifier] Cannot send email: not configured")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = self.recipient_email

            # Plain text version
            if body_text is None:
                # Simple HTML to text conversion
                import re
                body_text = re.sub('<[^<]+?>', '', body_html)
                body_text = body_text.replace('&nbsp;', ' ')

            msg.attach(MIMEText(body_text, 'plain'))
            msg.attach(MIMEText(body_html, 'html'))

            # Connect and send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, self.recipient_email, msg.as_string())

            print(f"[EmailNotifier] Email sent to {self.recipient_email}: {subject}")
            return True

        except Exception as e:
            print(f"[EmailNotifier] Error sending email: {e}")
            return False

    def send_alert(self, title: str, message: str) -> bool:
        """Send a simple alert email."""
        subject = f"[AutoTrade Alert] {title}"

        body_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #333;">{title}</h2>
            <p style="font-size: 14px; color: #666;">{message}</p>
            <hr style="border: 1px solid #eee;">
            <p style="font-size: 12px; color: #999;">
                Sent by AutoTrade at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """

        return self.send_email(subject, body_html)

    def send_rebalance_report(self, report: RebalanceReport) -> bool:
        """
        Send a detailed rebalancing report.

        Args:
            report: RebalanceReport object with all details

        Returns:
            True if email sent successfully.
        """
        subject = f"[AutoTrade] Rebalance Report - {report.strategy_name} - {report.timestamp.strftime('%Y-%m-%d')}"

        # Build HTML body
        body_html = self._build_rebalance_report_html(report)

        return self.send_email(subject, body_html)

    def _build_rebalance_report_html(self, report: RebalanceReport) -> str:
        """Build HTML content for rebalance report."""

        # Determine color based on risk level
        risk_colors = {
            'LOW': '#28a745',
            'WATCH': '#ffc107',
            'CAUTION': '#fd7e14',
            'ELEVATED': '#dc3545',
            'HIGH': '#dc3545',
            'HIGH RISK': '#dc3545',
            'EXTREME': '#6f42c1',
        }
        risk_color = risk_colors.get(report.risk_level, '#666')

        # Build actions table
        actions_html = ""
        if report.actions:
            actions_rows = ""
            for action in report.actions:
                action_color = '#28a745' if action.action == 'BUY' else '#dc3545'
                actions_rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">
                        <strong>{action.ticker}</strong>
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: {action_color};">
                        {action.action}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">
                        {action.shares:.0f}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">
                        ${action.price:.2f}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">
                        ${action.value:,.2f}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; font-size: 12px; color: #666;">
                        {action.reason}
                    </td>
                </tr>
                """

            actions_html = f"""
            <h3 style="color: #333; margin-top: 30px;">Actions Taken</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 10px; text-align: left;">Ticker</th>
                        <th style="padding: 10px; text-align: left;">Action</th>
                        <th style="padding: 10px; text-align: right;">Shares</th>
                        <th style="padding: 10px; text-align: right;">Price</th>
                        <th style="padding: 10px; text-align: right;">Value</th>
                        <th style="padding: 10px; text-align: left;">Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {actions_rows}
                </tbody>
            </table>
            """
        else:
            actions_html = """
            <h3 style="color: #333; margin-top: 30px;">Actions Taken</h3>
            <p style="color: #666;">No trades executed during this rebalance.</p>
            """

        # Build positions table
        positions_rows = ""
        sorted_positions = sorted(
            report.positions.items(),
            key=lambda x: x[1].get('value', 0),
            reverse=True
        )
        for ticker, pos_data in sorted_positions[:20]:  # Top 20
            weight = pos_data.get('weight', 0) * 100
            positions_rows += f"""
            <tr>
                <td style="padding: 6px; border-bottom: 1px solid #eee;"><strong>{ticker}</strong></td>
                <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: right;">
                    {pos_data.get('shares', 0):.0f}
                </td>
                <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: right;">
                    ${pos_data.get('value', 0):,.2f}
                </td>
                <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: right;">
                    {weight:.1f}%
                </td>
            </tr>
            """

        if len(report.positions) > 20:
            positions_rows += f"""
            <tr>
                <td colspan="4" style="padding: 6px; color: #999; font-style: italic;">
                    ... and {len(report.positions) - 20} more positions
                </td>
            </tr>
            """

        # Regime change indicator
        regime_change_html = ""
        if report.regime_change:
            regime_change_html = """
            <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 4px;">
                <strong>Regime Change Detected</strong> - Allocation has been adjusted.
            </div>
            """

        body_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto;">

            <h1 style="color: #333; border-bottom: 2px solid #333; padding-bottom: 10px;">
                Rebalance Report
            </h1>

            <p style="color: #666;">
                <strong>Strategy:</strong> {report.strategy_name}<br>
                <strong>Date:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            </p>

            {regime_change_html}

            <!-- Regime Status -->
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin: 0 0 10px 0; color: #333;">Market Regime</h3>
                <table style="width: 100%;">
                    <tr>
                        <td style="padding: 5px 0;">
                            <strong>Bear Score:</strong>
                        </td>
                        <td style="padding: 5px 0;">
                            {report.bear_score:.1f} / 100
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;">
                            <strong>Risk Level:</strong>
                        </td>
                        <td style="padding: 5px 0;">
                            <span style="color: {risk_color}; font-weight: bold;">
                                {report.risk_level}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;">
                            <strong>Allocation:</strong>
                        </td>
                        <td style="padding: 5px 0;">
                            {report.allocation_aggressive*100:.0f}% Aggressive /
                            {report.allocation_defensive*100:.0f}% Defensive
                        </td>
                    </tr>
                </table>
            </div>

            <!-- Reason -->
            <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin: 0 0 10px 0; color: #0066cc;">Rebalance Reason</h3>
                <p style="margin: 0; color: #333;">{report.rebalance_reason}</p>
            </div>

            {actions_html}

            <!-- Portfolio Summary -->
            <h3 style="color: #333; margin-top: 30px;">Portfolio Summary</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <table style="width: 100%;">
                    <tr>
                        <td style="padding: 5px 0;"><strong>Total Value:</strong></td>
                        <td style="padding: 5px 0; text-align: right; font-size: 18px;">
                            <strong>${report.portfolio_value:,.2f}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;"><strong>Cash:</strong></td>
                        <td style="padding: 5px 0; text-align: right;">${report.cash:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px 0;"><strong>Positions:</strong></td>
                        <td style="padding: 5px 0; text-align: right;">{len(report.positions)}</td>
                    </tr>
                </table>
            </div>

            <!-- Current Holdings -->
            <h3 style="color: #333; margin-top: 30px;">Current Holdings</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 8px; text-align: left;">Ticker</th>
                        <th style="padding: 8px; text-align: right;">Shares</th>
                        <th style="padding: 8px; text-align: right;">Value</th>
                        <th style="padding: 8px; text-align: right;">Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {positions_rows}
                </tbody>
            </table>

            <hr style="border: 1px solid #eee; margin-top: 30px;">
            <p style="font-size: 12px; color: #999;">
                This is an automated report from AutoTrade Regime Adaptive Strategy.<br>
                Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>

        </body>
        </html>
        """

        return body_html


def create_email_notifier(recipient_email: str) -> EmailNotifier:
    """
    Factory function to create an email notifier.

    Configures SMTP settings from environment variables.
    For Gmail, you need to:
    1. Enable 2-Step Verification
    2. Create an App Password at https://myaccount.google.com/apppasswords
    3. Set environment variables:
       - SMTP_USERNAME=your_email@gmail.com
       - SMTP_PASSWORD=your_app_password
    """
    return EmailNotifier(recipient_email=recipient_email)
