from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from base64 import b64encode, b64decode

def encrypt_aes(key, plaintext):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=backend)

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(plaintext) + padder.finalize()

    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    code = b64encode(ciphertext).decode('utf-8')
    print(f'Mã hoá AES: {code}')
    return code

def decrypt_aes(key, ciphertext):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=backend)

    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(b64decode(ciphertext)) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(decrypted_data) + unpadder.finalize()
    code = plaintext.decode('utf-8')
    print(f'Giải mã AES: {code}')
    return code

# Thử nghiệm mã hoá và giải mã
# key = b'ThisIsA16ByteKey'
# plaintext = 'Hello, AES!'

# ciphertext = encrypt_aes(key, plaintext.encode('utf-8'))
# print(f'Ciphertext: {ciphertext}')

# decrypted_text = decrypt_aes(key, ciphertext)
# print(f'Decrypted Text: {decrypted_text}')