from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'         
app.secret_key = 'supersecretkey'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o777)

# Ensure the static/uploads folder exists for keypoints
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads', mode=0o777)

# Function to preprocess image
def preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {path} could not be loaded.")
    
    # Resize to consistent size
    image = cv2.resize(image, (600, 600))

    # Convert to binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Complement the binary image
    complemented_image = cv2.bitwise_not(binary_image)

    # Crop to remove excess background
    x, y, w, h = cv2.boundingRect(complemented_image)
    cropped_image = complemented_image[y:y + h, x:x + w]

    return cropped_image

# Function to draw keypoints and save the result
def draw_and_save_keypoints(image, keypoints, filename):
    output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(filename, output_image)

# Function to align images based on keypoint matching
def align_images(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return img2  # Return original if no descriptors are found

    # BFMatcher with cross-check for best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned_img = cv2.warpPerspective(img2, matrix, (img1.shape[1], img1.shape[0]))
        return aligned_img
    return img2

# Function to calculate histogram similarity
def histogram_similarity(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity * 100

# Function to match images using SSIM, SIFT, and Histogram similarity
def match(path1, path2):
    img1 = preprocess_image(path1)
    img2 = preprocess_image(path2)

    # Align img2 to img1
    img2_aligned = align_images(img1, img2)

    # Structural Similarity Index (SSIM)
    similarity_value_ssim, _ = compare_ssim(img1, img2_aligned, full=True)
    similarity_value_ssim *= 100  # Convert to percentage

    # SIFT feature matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2_aligned, None)

    # Save keypoints images
    keypoints1_path = os.path.join('static/uploads', 'keypoints1.jpg')
    keypoints2_path = os.path.join('static/uploads', 'keypoints2.jpg')
    draw_and_save_keypoints(img1, kp1, keypoints1_path)
    draw_and_save_keypoints(img2_aligned, kp2, keypoints2_path)

    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        similarity_value_sift = len(good_matches) / max(len(kp1), len(kp2)) * 100
    else:
        similarity_value_sift = 0

    # Histogram similarity
    similarity_value_hist = histogram_similarity(img1, img2_aligned)

    # Weighted average of SSIM, SIFT, and Histogram similarity values
    similarity_value = (0.4 * similarity_value_ssim +
                        0.2 * similarity_value_sift +
                        0.4 * similarity_value_hist)
    return float(similarity_value)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/credits')
def credits():
    return render_template('credits.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'file1' not in request.files or 'file2' not in request.files:
        flash("Please upload two images to compare.")
        return redirect(url_for('index'))

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        flash("Please select both images.")
        return redirect(url_for('index'))

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))

    # Save files to the server
    file1.save(path1)
    file2.save(path2)

    # Perform matching
    similarity_value = match(path1, path2)

    # Determine result classification based on similarity percentage
    if similarity_value >= 85:
        result_text = "Genuine"
        color = "green"
    elif 50 <= similarity_value < 85:
        result_text = "Possibly Genuine"
        color = "orange"
    else:
        result_text = "Likely Forged"
        color = "red"

    # Render the result page with similarity percentage and classification
    return render_template('result.html', similarity=result_text, color=color, accuracy=similarity_value)

if __name__ == '__main__':
    app.run(debug=True)