import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import hashlib


data = "YourInputData"
sha256_hash = hashlib.sha256(data.encode()).hexdigest()

# Initialize an empty list to store the 2-character blocks
k_values = []

# Iterate through the characters of the SHA-256 hash with a step of 2
for i in range(0, len(sha256_hash), 2):
    # Extract a 2-character block and append it to the list
    k_values.append(sha256_hash[i:i+2])

# Convert each 2-character block in k_values to integer
k_int_values = [int(k, 16) for k in k_values]


c1 = 0.6723
c2 = 0.1839
c3 = 0.2343
c4 = 0.0982
c5 = 0.7521
c6 = 0.2567

# Calculating intermediate parameters - h1 to h6
# Perform XOR operation on k_int_values
result_xor1 = c1+ (k_int_values[0] ^ k_int_values[1] ^ k_int_values[2] ^ k_int_values[3] ^ k_int_values[4])
result_xor2 = c2+ (k_int_values[5] ^ k_int_values[6] ^ k_int_values[7] ^ k_int_values[8] ^ k_int_values[9])
result_xor3 = c3+ (k_int_values[10] ^ k_int_values[11] ^ k_int_values[12] ^ k_int_values[13] ^ k_int_values[14])
result_xor4 = c4+ (k_int_values[15] ^ k_int_values[16] ^ k_int_values[17] ^ k_int_values[18] ^ k_int_values[19])
result_xor5 = c5+ (k_int_values[20] ^ k_int_values[21] ^ k_int_values[22] ^ k_int_values[23] ^ k_int_values[24] ^ k_int_values[25] )
result_xor6 = c6 + (k_int_values[26] ^ k_int_values[27] ^ k_int_values[28] ^ k_int_values[29] ^ k_int_values[30] ^ k_int_values[31])

# Divide the result by 256
h1 = result_xor1 / 256
h2 = result_xor2 / 256
h3 = result_xor3 / 256
h4 = result_xor4 / 256
h5 = result_xor5 / 256
h6 = result_xor6 / 256


# Compute the initial values for the chaotic maps
x0 = np.mod((h1 + h2 + h5) * 10**8, 256) / 255
y0 = np.mod((h3 + h4 + h6) * 10**8, 256) / 255
z0 = np.mod((h1 + h2 + h3 + h4) * 10**8, 256) / 255
p0 = np.mod((h1 + h2 + h3) * 10**8, 256) / 255
q0 = np.mod((h4 + h5 + h6) * 10**8, 256) / 255


# Calculate the control parameters for the chaotic maps
sum_h = h1 + h2 + h3 + h4 + h5 + h6
a = np.mod((h1 + h2 / sum_h) * 100, 3) + 1
b = np.mod((h3 + h4 / sum_h) * 100, 3) + 1
c = np.mod((h5 + h6 / sum_h) * 100, 3) + 1
mu = np.mod((h1 + h2 + h3) / sum_h, 1)

# Function to iterate the 3D Sine chaotic system
def iterate_3d_sine_chaos(X0, Y0, Z0, a, b, c, num_iterations):
    X = [X0]
    Y = [Y0]
    Z = [Z0]

    for _ in range(num_iterations - 1):
        X_n = np.sin(a * Y[-1]) + c * np.sin(a * X[-1])
        Y_n = np.sin(b * Z[-1]) + c * np.sin(b * Y[-1])
        Z_n = np.sin(c * X[-1]) + a * np.sin(c * Z[-1])

        X.append(X_n)
        Y.append(Y_n)
        Z.append(Z_n)

    return np.array(X), np.array(Y), np.array(Z)

# Function to iterate the 2D LASM chaotic system
def iterate_2d_lasm_chaos(Y0, Z0, p, q, mu, num_iterations):
    Y = [Y0]
    Z = [Z0]

    for _ in range(num_iterations - 1):
        Y_n = np.mod(mu * Y[-1] * (1 - Y[-1]), 1)
        Z_n = np.mod(p * Z[-1] + q * Y[-1], 1)

        Y.append(Y_n)
        Z.append(Z_n)

    return np.array(Y), np.array(Z)

def generate_Z_sequence(Z_prime, u):
    # Calculate the length of the Z sequence based on u^2/8
    Z_length = int(u**2 / 8)

    # Take the first Z_length values from Z_prime to generate Z
    Z = Z_prime[:Z_length]

    return Z


image_path = 'less.png'
image = Image.open(image_path)

# Convert the RGB image to grayscale image
gray_image = image.convert('L')

# Convert to numpy array and then to 8-bit binary
gray_array = np.array(gray_image)


# Create bit planes
bit_planes = np.unpackbits(np.expand_dims(gray_array, axis=-1), axis=-1)
# Reshape the bit planes to form a 3D array
bit_planes_3d = bit_planes.reshape(gray_array.shape + (8,))


# Stack all bit planes horizontally to form a 3D cube
bit_cube = np.concatenate([bit_planes_3d[:, :, 7 - bit][:, :, np.newaxis] for bit in range(8)], axis=-1)

# Display each bit plane as a subplot
fig, axes = plt.subplots(1, 8, figsize=(20, 3))

for i, ax in enumerate(axes):
    ax.imshow(bit_cube[:, :, i], cmap='gray')
    ax.set_title(f"Bit Plane {8 - i}")
    ax.axis('off')
    
# Shuffle the binary values in the 3D matrix
shuffled_bit_planes_3d = 1 - bit_planes_3d

# Number of iterations
num_iterations = bit_planes_3d.shape[0] * bit_planes_3d.shape[1]

# Perform iteration of the 3D Sine chaotic system
X1, X2, X3 = iterate_3d_sine_chaos(x0, y0, z0, a, b, c, num_iterations)

# Reshape X1, X2, and X3 to match the shape of bit_planes_3d
X1 = X1.reshape(bit_planes_3d.shape[0], bit_planes_3d.shape[1])
X2 = X2.reshape(bit_planes_3d.shape[0], bit_planes_3d.shape[1])
X3 = X3.reshape(bit_planes_3d.shape[0], bit_planes_3d.shape[1])

# Convert X1, X2, and X3 to 8-bit binary and store in a 3D array
X1_binary = np.unpackbits(X1.astype(np.uint8).reshape(X1.shape + (1,)), axis=-1)
X2_binary = np.unpackbits(X2.astype(np.uint8).reshape(X2.shape + (1,)), axis=-1)
X3_binary = np.unpackbits(X3.astype(np.uint8).reshape(X3.shape + (1,)), axis=-1)


# Perform XOR operation on X1_binary and X2_binary
x_seq_result = np.bitwise_xor(np.bitwise_xor(X1_binary, X2_binary),X3_binary)

new_bit_planes = np.bitwise_xor(shuffled_bit_planes_3d,x_seq_result)


encoding_rule = {
    '00': 'A',
    '11': 'T',
    '01': 'C',
    '10': 'G'
}

# Apply the encoding rule to the entire 3D array
def map_to_dna(bit_planes_3d):
    # Convert 3D array to 2D array for easy processing
    flattened_array = bit_planes_3d.reshape(-1, 8)

    # Convert each 8-bit binary value to DNA sequence
    dna_sequences = []
    for binary_value in flattened_array:
        binary_string = ''.join(map(str, binary_value))
        binary_pairs = [binary_string[i:i+2] for i in range(0, len(binary_string), 2)]
        dna_sequence = ''.join(encoding_rule[pair] for pair in binary_pairs)
        dna_sequences.append(dna_sequence)

    # Reshape the result back to the original 3D shape
    mapped_dna_array = np.array(dna_sequences).reshape(bit_planes_3d.shape[:-1])

    return mapped_dna_array

# Apply DNA mapping to the shuffled bit planes
mapped_dna_array = map_to_dna(new_bit_planes)


# Define the transcription rule
transcription_rule = {
    'A': 'U',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}

# Apply DNA transcription to the mapped DNA array
def transcribe_dna(mapped_dna_array):
    # Iterate over each element in the array and apply transcription rule
    transcribed_array = np.vectorize(lambda x: ''.join(transcription_rule[n] for n in x))(mapped_dna_array)

    return transcribed_array

# Apply DNA transcription to the mapped DNA array
transcribed_dna_array = transcribe_dna(mapped_dna_array)


# Number of iterations
num_iterations_2d_lasm =  bit_planes_3d.shape[0] * bit_planes_3d.shape[1]

# Perform iteration of the 2D LASM chaotic system
Y, Z_prime = iterate_2d_lasm_chaos(y0, z0, p0, q0, mu, num_iterations_2d_lasm)

Y = Y.reshape(bit_planes_3d.shape[0],bit_planes_3d.shape[1])
Z_prime = Z_prime.reshape(bit_planes_3d.shape[0],bit_planes_3d.shape[1])


# Scale Y and Z to the range [0, 255]
scaled_Y = (Y * 255).astype(np.uint8)
scaled_Z_prime = (Z_prime * 255).astype(np.uint8)

# Convert scaled Y and Z to 8-bit binary and store in a 3D array
Y_binary = np.unpackbits(scaled_Y.reshape(scaled_Y.shape + (1,)), axis=-1)
Z_prime_binary = np.unpackbits(scaled_Z_prime.reshape(scaled_Z_prime.shape + (1,)), axis=-1)


# Define the encoding rule
rna_mutation_step_1 = {
    '00': 'A',
    '11': 'U',
    '01': 'G',
    '10': 'C'
}

# Apply the encoding rule to the entire 3D array
def first_mutation(y_binary):
    # Convert 3D array to 2D array for easy processing
    flattened_array = y_binary.reshape(-1, 8)

    # Convert each 8-bit binary value to DNA sequence
    first_mutation_seq = []
    for binary_value in flattened_array:
        binary_string = ''.join(map(str, binary_value))
        binary_pairs = [binary_string[i:i+2] for i in range(0, len(binary_string), 2)]
        sequence = ''.join(rna_mutation_step_1[pair] for pair in binary_pairs)
        first_mutation_seq.append(sequence)

    # Reshape the result back to the original 3D shape
    mapped_rna1_array = np.array(first_mutation_seq).reshape(y_binary.shape[:-1])

    return mapped_rna1_array

# Apply DNA mapping to the shuffled bit planes
mapped_rna1_array = first_mutation(Y_binary)


# Define the RNA mutation rule
rna_mutation_rule = {
    'A': 'G',
    'U': 'C',
    'G': 'A',
    'C': 'U'
}

# Apply RNA mutation to the translated RNA array
def mutate_rna(rna_array):
    # Iterate over each element in the array and apply RNA mutation rule
    mutated_rna_array = np.vectorize(lambda x: ''.join(rna_mutation_rule[n] for n in x))(rna_array)

    return mutated_rna_array

# Apply RNA mutation to the mapped_rna_array
mutated_rna_array = mutate_rna(mapped_rna1_array)


# Define the RNA translation
rna_translation_rule = {
    'A': 'U', 
    'U': 'G', 
    'G': 'C', 
    'C': 'A'
    }
# Function to apply RNA translation
def translate_rna(rna_array):
    translated_rna_array = np.vectorize(lambda x: ''.join(rna_translation_rule[n] for n in x))(rna_array)

    return translated_rna_array

# Apply translation
translated_rna_array = translate_rna(mapped_rna1_array)

encryption_rules = {
  'A':'00',
  'U':'11',
  'G':'01',
  'C':'10'
}
# Initialize output 3D array
binary_array = np.zeros((len(translated_rna_array), len(translated_rna_array[0]), len(translated_rna_array[0][0])*2), dtype=int)

# Map each base to binary
for i in range(len(translated_rna_array)):
  for j in range(len(translated_rna_array[i])):
    for k, base in enumerate(translated_rna_array[i][j]):
      binary_array[i,j,k*2:k*2+2] = [int(x) for x in encryption_rules[base]]
      

Z = Z_prime[:int(256/8)]

# Scale Y and Z to the range [0, 255]
scaled_Z = (Z * 255).astype(np.uint8)
Z_binary = np.unpackbits(scaled_Z.reshape(scaled_Z.shape + (1,)), axis=-1)

# Select the first 32 elements along the first axis
selected_Z_binary = Z_binary[:32]

# Select the first 32 elements along the first axis of binary_array
selected_binary_array = binary_array[:32]

# Perform XOR operation only for the first 32 elements
z_result = np.bitwise_xor(selected_Z_binary, selected_binary_array)

# Select the first 32 elements along the first axis
selected_binary_array = binary_array.copy()  # Make a copy to avoid modifying the original array
selected_binary_array[:32] = z_result

# Now, encrypted_array has the first 32 elements replaced with z_result
encrypted_array = selected_binary_array

# Number of rows and columns in the binary_array
num_rows, num_cols, num_channels = encrypted_array.shape

# Initialize random 3D binary array
encrypted_array = np.random.randint(0, 2, size=(num_rows, num_cols, num_channels))

# Get dimensions
rows, cols, channels = encrypted_array.shape

# Create empty image array
image = np.zeros((rows, cols))

# Populate image by summing values across channels
for i in range(rows):
  for j in range(cols):
    image[i,j] = np.sum(encrypted_array[i,j,:])

# Normalize to 0-255 range
image = image - np.min(image)
image = (255*image/np.max(image)).astype(np.uint8)

# Save encrypted image 
plt.imsave('encrypted_image.png', image, cmap='gray')



## DECRYPTION STARTS HERE :

# Flatten the 3D binary array
flattened_binary_array = encrypted_array.reshape(-1)

# Normalize the flattened array to [0, 1]
normalized_binary_array = flattened_binary_array.astype(float) / 255.0

# Threshold the normalized array to get the binary values
flat_binary_array = (normalized_binary_array > 0.5).astype(int)

# Ensure the size of flat_binary_array matches the size of encrypted_array
assert flat_binary_array.size == np.prod(encrypted_array.shape), f"Size mismatch: {flat_binary_array.size} vs {np.prod(encrypted_array.shape)}"

# Reshape the flat binary array to the original 3D shape
decrypted_array = flat_binary_array.reshape(encrypted_array.shape)


# Reverse the XOR operation for the first 32 elements
decryption_binary_array = np.bitwise_xor(selected_Z_binary, z_result)

# Replace the first 32 elements in the selected binary array with the reversed decryption result
decrypted_binary_array = selected_binary_array.copy()
decrypted_binary_array[:32] = decryption_binary_array

# Create an empty 3D array for the decrypted RNA
decrypted_rna_array = np.zeros((len(decrypted_binary_array), len(decrypted_binary_array[0]), len(decrypted_binary_array[0][0]) // 2), dtype='U1')

# Map each binary pair to RNA bases
for i in range(len(decrypted_binary_array)):
    for j in range(len(decrypted_binary_array[i])):
        for k in range(0, len(decrypted_binary_array[i][j]), 2):
            pair = ''.join(map(str, decrypted_binary_array[i, j, k:k+2]))
            for base, binary_pair in encryption_rules.items():
                if binary_pair == pair:
                    decrypted_rna_array[i, j, k // 2] = base


# Define the reverse RNA translation rule
reverse_translated_rna_rules = {'U': 'A', 'G': 'U', 'C': 'G', 'A': 'C'}

# Function to apply reverse RNA translation
def reverse_translate_rna(rna_array):
    reverse_translated_rna_array = np.vectorize(lambda x: ''.join(reverse_translated_rna_rules[n] for n in x))(rna_array)
    return reverse_translated_rna_array

# Apply reverse translation
reverse_translated_rna_array = reverse_translate_rna(translated_rna_array)


reverse_translated_rna_rules = {'U': 'A', 'G': 'U', 'C': 'G', 'A': 'C'}

def reverse_translate_rna(rna_array):
    return np.vectorize(lambda x: ''.join(reverse_translated_rna_rules[n] for n in x))(rna_array)

# Apply reverse translation to the translated RNA array
reverse_translated_rna_array = reverse_translate_rna(translated_rna_array)

# Now, define and apply the reverse mutation rule to the reverse_translated_rna_array
reverse_rna_mutation_rule = {'G': 'A', 'C': 'U', 'A': 'G', 'U': 'G'}

def reverse_mutate_rna(rna_array):
    return np.vectorize(lambda x: ''.join(reverse_rna_mutation_rule[n] for n in x))(rna_array)

# Apply reverse RNA mutation
reversed_mutated_rna_array = reverse_mutate_rna(reverse_translated_rna_array)


# Define the reverse encoding rule for the first mutation
reverse_rna_mutation_step_1 = {
    'A': '00', 'U': '11','C': '01','G': '10'
}

# Apply the reverse encoding rule for the first mutation
def reverse_first_mutation(mapped_rna1_array):
    # Convert 3D array to 1D array for easy processing
    flattened_array = mapped_rna1_array.flatten()

    # Convert each RNA sequence to 8-bit binary values
    binary_sequences = []
    for rna_sequence in flattened_array:
        binary_values = [reverse_rna_mutation_step_1[base] for base in rna_sequence]
        binary_string = ''.join(binary_values)
        binary_sequence = [int(bit) for bit in binary_string]
        binary_sequences.append(binary_sequence)

    # Reshape the result back to the original 3D shape
    reversed_Y_binary = np.array(binary_sequences).reshape(mapped_rna1_array.shape + (8,))

    return reversed_Y_binary

# Apply the reverse DNA mapping for the first mutation
reversed_Y_binary = reverse_first_mutation(mapped_rna1_array)

# Update the DNA transcription rule to handle RNA-specific bases
dna_transcription_rule = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C',
    'U': 'A'  # Assuming 'U' in RNA maps to 'A' in DNA
}

# Apply reverse DNA transcription to the reversed mutated RNA array
def reverse_transcribe_dna(reversed_mutated_rna_array):
    # Replace 'U' with 'T' to convert RNA back to DNA
    corrected_rna_array = np.vectorize(lambda x: x.replace('U', 'T'))(reversed_mutated_rna_array)

    # Iterate over each element in the corrected array and apply reverse DNA transcription rule
    reversed_transcription_array = np.vectorize(lambda x: ''.join(dna_transcription_rule[n] for n in x))(corrected_rna_array)

    return reversed_transcription_array

# Apply reverse DNA transcription to the reversed mutated RNA array
reversed_dna_transcription_array = reverse_transcribe_dna(reversed_mutated_rna_array)


# Define the decoding rule (inverse of encoding_rule)
decoding_rule = {
    'A': '00',
    'T': '11',
    'C': '01',
    'G': '10'
}

# Apply the decoding rule to the entire 3D array
def map_to_bits(dna_array_3d):
    # Flatten the array for easier processing
    flattened_dna_array = dna_array_3d.reshape(-1)

    # Convert each DNA sequence back to binary values
    binary_values = []
    for dna_sequence in flattened_dna_array:
        binary_string = ''.join(decoding_rule[nucleotide] for nucleotide in dna_sequence)
        binary_array = np.array([int(bit) for bit in binary_string])
        binary_values.append(binary_array)

    # Reshape the result back to the original 3D shape of bit planes
    decrypted_bit_planes_3d = np.array(binary_values).reshape(dna_array_3d.shape + (8,))

    return decrypted_bit_planes_3d

# Apply DNA to bit plane mapping to get the decrypted bit planes
decrypted_bit_planes_3d = map_to_bits(mapped_dna_array)


# Recompute X1, X2, and X3
X1, X2, X3 = iterate_3d_sine_chaos(x0, y0, z0, a, b, c, num_iterations)

# Reshape X1, X2, and X3 to match the shape
X1 = X1.reshape(bit_planes_3d.shape[0], bit_planes_3d.shape[1])
X2 = X2.reshape(bit_planes_3d.shape[0], bit_planes_3d.shape[1])
X3 = X3.reshape(bit_planes_3d.shape[0], bit_planes_3d.shape[1])

# Convert X1, X2, and X3 to 8-bit binary
X1_binary = np.unpackbits(X1.astype(np.uint8).reshape(X1.shape + (1,)), axis=-1)
X2_binary = np.unpackbits(X2.astype(np.uint8).reshape(X2.shape + (1,)), axis=-1)
X3_binary = np.unpackbits(X3.astype(np.uint8).reshape(X3.shape + (1,)), axis=-1)

# Perform XOR operation on X1_binary, X2_binary, and X3_binary
x_seq_result = np.bitwise_xor(np.bitwise_xor(X1_binary, X2_binary),X3_binary)

# Perform XOR operation on new_bit_planes and x_seq_result to get the original shuffled_bit_planes_3d
reversed_shuffled_bit_planes_3d = np.bitwise_xor(new_bit_planes, x_seq_result)

# Function to unscramble the matrix by flipping 1s to 0s and vice versa of binary values in the 3D matrix
unshuffled_bit_planes_3d = 1 - shuffled_bit_planes_3d

# Function to convert a binary matrix to pixel values
def binary_to_pixels(binary_matrix):
    # Reshape the binary matrix to a flat 1D array
    flat_binary = binary_matrix.reshape(-1, 8)

    # Convert each 8-bit binary value to an integer
    pixel_values = []
    for binary_value in flat_binary:
        binary_string = ''.join(map(str, binary_value))
        decimal_value = int(binary_string, 2)
        pixel_values.append(decimal_value)

    # Reshape the result back to the original 2D shape (image dimensions)
    pixel_array = np.array(pixel_values, dtype=np.uint8).reshape(binary_matrix.shape[:-1])

    return pixel_array

# Apply the conversion to pixel values
pixel_image = binary_to_pixels(unshuffled_bit_planes_3d)

# Save decrypted image 
plt.imsave('decrypted_image.png', pixel_image, cmap='gray')