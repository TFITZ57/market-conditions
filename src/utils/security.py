import os
import logging
import json
import hashlib
import base64
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_api_key(service):
    """
    Get API key for a specific service.
    
    Args:
        service (str): Service name ('fred', 'bls', 'attom', 'openai', 'anthropic').
    
    Returns:
        str: API key or None if not found.
    """
    try:
        # Map service name to environment variable
        service_to_env = {
            'fred': 'FRED_API_KEY',
            'bls': 'BLS_API_KEY',
            'attom': 'ATTOM_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        
        # Get environment variable name
        env_var = service_to_env.get(service.lower())
        
        if not env_var:
            logger.error(f"Unknown service: {service}")
            return None
        
        # Get API key from environment
        api_key = os.getenv(env_var)
        
        if not api_key:
            logger.warning(f"API key not found for service: {service} (environment variable: {env_var})")
            return None
        
        return api_key
    
    except Exception as e:
        logger.error(f"Error getting API key for {service}: {str(e)}")
        return None

def mask_api_key(api_key, visible_chars=4):
    """
    Mask an API key for display.
    
    Args:
        api_key (str): API key to mask.
        visible_chars (int, optional): Number of characters to show at start and end. Defaults to 4.
    
    Returns:
        str: Masked API key.
    """
    if not api_key:
        return None
    
    # Ensure visible_chars is reasonable
    visible_chars = min(visible_chars, len(api_key) // 3)
    
    # Mask the API key
    if len(api_key) <= visible_chars * 2:
        # Key is too short to mask meaningfully
        return '*' * len(api_key)
    
    return api_key[:visible_chars] + '*' * (len(api_key) - visible_chars * 2) + api_key[-visible_chars:]

def generate_key_from_password(password, salt=None):
    """
    Generate a key from a password for encryption/decryption.
    
    Args:
        password (str): Password to derive key from.
        salt (bytes, optional): Salt for key derivation. Defaults to None.
    
    Returns:
        tuple: (key, salt) tuple.
    """
    try:
        # Convert password to bytes
        password_bytes = password.encode()
        
        # Generate salt if not provided
        if salt is None:
            salt = os.urandom(16)
        
        # Generate key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        
        return key, salt
    
    except Exception as e:
        logger.error(f"Error generating key from password: {str(e)}")
        return None, None

def encrypt_data(data, key):
    """
    Encrypt data using Fernet symmetric encryption.
    
    Args:
        data (str or dict): Data to encrypt.
        key (bytes): Encryption key.
    
    Returns:
        bytes: Encrypted data.
    """
    try:
        # Convert data to string if it's a dictionary
        if isinstance(data, dict):
            data = json.dumps(data)
        
        # Convert to bytes if needed
        if isinstance(data, str):
            data = data.encode()
        
        # Create Fernet cipher
        cipher = Fernet(key)
        
        # Encrypt data
        encrypted_data = cipher.encrypt(data)
        
        return encrypted_data
    
    except Exception as e:
        logger.error(f"Error encrypting data: {str(e)}")
        return None

def decrypt_data(encrypted_data, key):
    """
    Decrypt data using Fernet symmetric encryption.
    
    Args:
        encrypted_data (bytes): Encrypted data.
        key (bytes): Decryption key.
    
    Returns:
        str or dict: Decrypted data.
    """
    try:
        # Create Fernet cipher
        cipher = Fernet(key)
        
        # Decrypt data
        decrypted_data = cipher.decrypt(encrypted_data)
        
        # Convert to string
        decrypted_str = decrypted_data.decode()
        
        # Try to parse as JSON
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError:
            # Return as string if not valid JSON
            return decrypted_str
    
    except Exception as e:
        logger.error(f"Error decrypting data: {str(e)}")
        return None

def secure_hash(data, salt=None):
    """
    Create a secure hash of data.
    
    Args:
        data (str): Data to hash.
        salt (bytes, optional): Salt for hashing. Defaults to None.
    
    Returns:
        tuple: (hash, salt) tuple.
    """
    try:
        # Convert data to bytes if needed
        if isinstance(data, str):
            data = data.encode()
        
        # Generate salt if not provided
        if salt is None:
            salt = os.urandom(16)
        
        # Create hash
        hash_obj = hashlib.sha256()
        hash_obj.update(salt)
        hash_obj.update(data)
        
        # Get hash digest
        hash_digest = hash_obj.hexdigest()
        
        return hash_digest, salt
    
    except Exception as e:
        logger.error(f"Error creating secure hash: {str(e)}")
        return None, None

def verify_hash(data, hash_digest, salt):
    """
    Verify that data matches a hash.
    
    Args:
        data (str): Data to verify.
        hash_digest (str): Expected hash.
        salt (bytes): Salt used for hashing.
    
    Returns:
        bool: True if verified, False otherwise.
    """
    try:
        # Convert data to bytes if needed
        if isinstance(data, str):
            data = data.encode()
        
        # Create hash
        hash_obj = hashlib.sha256()
        hash_obj.update(salt)
        hash_obj.update(data)
        
        # Get hash digest
        calculated_hash = hash_obj.hexdigest()
        
        # Compare hashes
        return calculated_hash == hash_digest
    
    except Exception as e:
        logger.error(f"Error verifying hash: {str(e)}")
        return False

def store_api_key(service, api_key, password, key_file=None):
    """
    Securely store an API key.
    
    Args:
        service (str): Service name.
        api_key (str): API key to store.
        password (str): Password for encryption.
        key_file (str, optional): Path to key file. Defaults to None.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Set default key file if not provided
        if key_file is None:
            key_dir = os.path.join(os.path.expanduser('~'), '.fairfield_housing')
            os.makedirs(key_dir, exist_ok=True)
            key_file = os.path.join(key_dir, 'api_keys.enc')
        
        # Generate key from password
        key, salt = generate_key_from_password(password)
        
        if key is None:
            return False
        
        # Load existing keys if file exists
        keys_data = {}
        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # Decrypt existing data
                keys_data = decrypt_data(encrypted_data, key)
                
                if keys_data is None:
                    logger.error("Failed to decrypt existing keys. Password may be incorrect.")
                    return False
                
                # If keys_data is not a dictionary, something is wrong
                if not isinstance(keys_data, dict):
                    logger.error("Decrypted data is not a dictionary.")
                    return False
            
            except Exception as e:
                logger.error(f"Error loading existing keys: {str(e)}")
                # Start with empty dictionary if there's an error
                keys_data = {}
        
        # Add or update the API key
        keys_data[service.lower()] = api_key
        
        # Encrypt the updated keys
        encrypted_keys = encrypt_data(keys_data, key)
        
        if encrypted_keys is None:
            return False
        
        # Write encrypted keys to file
        with open(key_file, 'wb') as f:
            f.write(encrypted_keys)
        
        logger.info(f"Stored API key for {service}")
        return True
    
    except Exception as e:
        logger.error(f"Error storing API key: {str(e)}")
        return False

def retrieve_api_key(service, password, key_file=None):
    """
    Retrieve a securely stored API key.
    
    Args:
        service (str): Service name.
        password (str): Password for decryption.
        key_file (str, optional): Path to key file. Defaults to None.
    
    Returns:
        str: API key or None if not found.
    """
    try:
        # Set default key file if not provided
        if key_file is None:
            key_dir = os.path.join(os.path.expanduser('~'), '.fairfield_housing')
            key_file = os.path.join(key_dir, 'api_keys.enc')
        
        # Check if key file exists
        if not os.path.exists(key_file):
            logger.warning(f"Key file not found: {key_file}")
            return None
        
        # Generate key from password
        key, _ = generate_key_from_password(password)
        
        if key is None:
            return None
        
        # Read encrypted keys
        with open(key_file, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt data
        keys_data = decrypt_data(encrypted_data, key)
        
        if keys_data is None:
            logger.error("Failed to decrypt keys. Password may be incorrect.")
            return None
        
        # If keys_data is not a dictionary, something is wrong
        if not isinstance(keys_data, dict):
            logger.error("Decrypted data is not a dictionary.")
            return None
        
        # Get API key for the service
        api_key = keys_data.get(service.lower())
        
        if api_key is None:
            logger.warning(f"API key not found for service: {service}")
        
        return api_key
    
    except Exception as e:
        logger.error(f"Error retrieving API key: {str(e)}")
        return None

def validate_api_key(service, api_key):
    """
    Validate an API key for a service.
    
    Args:
        service (str): Service name.
        api_key (str): API key to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    # This is a placeholder for actual API key validation
    # In a real implementation, this would make a test request to the API
    # to verify that the key is valid
    
    try:
        # Basic validation - check if key is not empty
        if not api_key:
            return False
        
        # Service-specific validation
        if service.lower() == 'fred':
            # FRED API keys are typically alphanumeric and 32 characters long
            return len(api_key) == 32 and api_key.isalnum()
        
        elif service.lower() == 'bls':
            # BLS API keys vary in format
            return len(api_key) > 8
        
        elif service.lower() == 'attom':
            # ATTOM API keys are typically longer alphanumeric strings
            return len(api_key) > 20
        
        elif service.lower() == 'openai':
            # OpenAI API keys start with "sk-" and are 51 characters long
            return api_key.startswith('sk-') and len(api_key) == 51
        
        elif service.lower() == 'anthropic':
            # Anthropic API keys start with "sk-ant-"
            return api_key.startswith('sk-ant-')
        
        else:
            logger.warning(f"Unknown service for API key validation: {service}")
            # Default to basic validation for unknown services
            return len(api_key) > 8
    
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return False
