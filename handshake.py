"""Create a Schwab OAuth refresh token without repeatedly entering app credentials."""

import argparse
import base64
import getpass
import json
import os
import secrets
import subprocess
import sys
import urllib.parse

try:
    import requests
except ImportError:
    print("[ERROR] Missing required library: requests")
    print("Install dependencies first with: python3 -m pip install -r requirements.txt")
    sys.exit(1)


KEYCHAIN_SERVICE = "rbob-fuel-tracker.schwab-oauth"
KEYCHAIN_FIELDS = ("app_key", "app_secret", "redirect_uri")


def keychain_account():
    return getpass.getuser()


def load_keychain_credentials():
    """Return saved credentials on macOS, or None when they have not been configured."""
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password", "-a", keychain_account(),
                "-s", KEYCHAIN_SERVICE, "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    try:
        credentials = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if all(isinstance(credentials.get(field), str) and credentials[field].strip()
           for field in KEYCHAIN_FIELDS):
        return credentials
    return None


def save_keychain_credentials(credentials):
    payload = json.dumps({field: credentials[field].strip() for field in KEYCHAIN_FIELDS})
    try:
        result = subprocess.run(
            [
                "security", "add-generic-password", "-U", "-a", keychain_account(),
                "-s", KEYCHAIN_SERVICE, "-w", payload,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "macOS Keychain is unavailable on this system"
    if result.returncode != 0:
        return False, result.stderr.strip() or "Keychain write failed"
    return True, None


def delete_keychain_credentials():
    try:
        result = subprocess.run(
            [
                "security", "delete-generic-password", "-a", keychain_account(),
                "-s", KEYCHAIN_SERVICE,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "macOS Keychain is unavailable on this system"
    if result.returncode != 0:
        return False, "No saved Schwab handshake credentials were found"
    return True, None


def environment_credentials():
    values = {
        "app_key": os.environ.get("SCHWAB_APP_KEY", "").strip(),
        "app_secret": os.environ.get("SCHWAB_APP_SECRET", "").strip(),
        "redirect_uri": os.environ.get("SCHWAB_REDIRECT_URI", "").strip(),
    }
    return values if all(values.values()) else None


def prompt_for_credentials():
    return {
        "app_key": input("Enter your Schwab App Key: ").strip(),
        "app_secret": getpass.getpass("Enter your Schwab App Secret: ").strip(),
        "redirect_uri": input(
            "Enter your Schwab App Redirect URI "
            "(usually https://127.0.0.1 or https://localhost): "
        ).strip(),
    }


def get_credentials(force_configure=False):
    if not force_configure:
        credentials = environment_credentials()
        if credentials:
            print("Using Schwab credentials from environment variables.")
            return credentials
        credentials = load_keychain_credentials()
        if credentials:
            print("Using saved Schwab credentials from macOS Keychain.")
            return credentials

    credentials = prompt_for_credentials()
    if not all(credentials.values()):
        raise ValueError("App key, app secret, and redirect URI are all required.")
    saved, error = save_keychain_credentials(credentials)
    if saved:
        print("Saved app credentials and redirect URI to macOS Keychain for future handshakes.")
    else:
        print(f"[WARNING] Could not save credentials to Keychain: {error}")
    return credentials


def authorization_url(credentials, state):
    params = {
        "response_type": "code",
        "client_id": credentials["app_key"],
        "redirect_uri": credentials["redirect_uri"],
        "state": state,
    }
    return f"https://api.schwabapi.com/v1/oauth/authorize?{urllib.parse.urlencode(params)}"


def parse_authorization_code(redirected_input, expected_state):
    code = redirected_input
    if "code=" in redirected_input:
        parsed = urllib.parse.urlparse(redirected_input)
        query = urllib.parse.parse_qs(parsed.query)
        returned_state = query.get("state", [None])[0]
        if returned_state and returned_state != expected_state:
            raise ValueError("OAuth state mismatch. Do not use this authorization code.")
        code = query.get("code", [None])[0]
    if not code:
        raise ValueError("Failed to extract an authorization code from the redirect URL.")
    return code


def exchange_code(credentials, code):
    auth = base64.b64encode(
        f"{credentials['app_key']}:{credentials['app_secret']}".encode()
    ).decode()
    response = requests.post(
        "https://api.schwabapi.com/v1/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": credentials["redirect_uri"],
        },
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_handshake(force_configure=False):
    credentials = get_credentials(force_configure)
    state = secrets.token_urlsafe(24)

    print("\n----------------------------------------------------------------------")
    print("STEP 1: AUTHORIZE THE APP IN YOUR BROWSER")
    print("----------------------------------------------------------------------")
    print("Open this URL in your browser:\n")
    print(authorization_url(credentials, state))
    print("\nAfter authorizing, paste the entire redirected URL here.")
    redirected_input = input("Redirected URL: ").strip()
    code = parse_authorization_code(redirected_input, state)

    print("\n----------------------------------------------------------------------")
    print("STEP 2: EXCHANGING CODE FOR REFRESH TOKEN")
    print("----------------------------------------------------------------------")
    tokens = exchange_code(credentials, code)
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Schwab responded successfully but did not return a refresh token.")
    print("\n[SUCCESS] Update the SCHWAB_REFRESH_TOKEN GitHub secret with this value:\n")
    print("======================================================================")
    print(refresh_token)
    print("======================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Create a Schwab OAuth refresh token using saved local credentials."
    )
    parser.add_argument(
        "--configure", action="store_true",
        help="Replace saved macOS Keychain app credentials and redirect URI.",
    )
    parser.add_argument(
        "--forget-credentials", action="store_true",
        help="Delete the saved macOS Keychain credentials without contacting Schwab.",
    )
    args = parser.parse_args()
    if args.configure and args.forget_credentials:
        parser.error("--configure and --forget-credentials cannot be used together")
    if args.forget_credentials:
        deleted, error = delete_keychain_credentials()
        print("Saved Schwab handshake credentials deleted." if deleted else f"[ERROR] {error}")
        return 0 if deleted else 1

    print("=" * 70)
    print("CHARLES SCHWAB API OAUTH HANDSHAKE")
    print("=" * 70)
    try:
        run_handshake(force_configure=args.configure)
    except (ValueError, RuntimeError, requests.RequestException) as error:
        print(f"[ERROR] {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
