import cv2
import numpy as np

# Detect game pice by color
# And create a rect around the detection
def find_bounding_rect(image, lower_color, upper_color):
    """    Detects the largest contour of a specified color in the image and returns its bounding rectangle.
    Args:
        image (numpy.ndarray): The input image in which to find the contour.
        lower_color (tuple): The lower bound of the color in HSV format.
        upper_color (tuple): The upper bound of the color in HSV format.
    Returns:
        tuple: A tuple containing the coordinates (x, y) and dimensions (width, height) of the bounding rectangle.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Combine masks to get only red pixels
    red_mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Threshold the mask to create a binary image
    _, thresh = cv2.threshold(red_mask, 127, 255, cv2.THRESH_BINARY)

    # Define the kernel for the morphological operation
    # ChatGPT gave me this part of the code, I have no idea how it works
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological opening to remove noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform morphological closing to close small holes
    # This really helps to cleanup the mask
    processed_img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.drawContours(image, cont, -1, (0, 255, 0), 3)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("Frame", image)
    cv2.imshow("Mask", processed_img)
    cv2.waitKey(0)

    return (x + w//2, y + h//2, w, h)
def find_matching_rows(df, target, start_tol=2, max_tol=15, step=2):
    """
    Find rows in the dataframe that match the target values within a dynamically
    increasing tolerance. If no rows are found for a very small tolerance, the
    tolerance will be increased until at least one row is found or the maximum
    tolerance is reached.

    Parameters:
        df (pd.DataFrame): DataFrame with the columns.
        target (dict): Target values for each column.
        start_tol (int): Starting tolerance measured in pixels.
        max_tol (int): Maximum allowed tolerance measured in pixels.
        step (int): Increment to increase tolerance on each iteration measured in pixels.

    Returns:
        filtered_df (pd.DataFrame): DataFrame with matching rows.
        used_tol (int): Tolerance at which the matching rows were found.
    """
    
    tolerance = start_tol
    while tolerance <= max_tol:
        # Create mask using np.isclose for each column
        mask = (
            np.isclose(df['Center_X'], target['Center_X'], atol=tolerance) &
            np.isclose(df['Center_Y'], target['Center_Y'], atol=tolerance) &
            np.isclose(df['Width'], target['Width'], atol=tolerance) &
            np.isclose(df['Height'], target['Height'], atol=tolerance)
        )
        filtered_df = df[mask]
        
        # Check if any row is found
        if not filtered_df.empty:
            print(f"Found rows with tolerance: {tolerance}")
            return filtered_df, tolerance
        
        # Increase the tolerance and try again
        tolerance += step

    # No rows found within maximum tolerance
    print("No rows found within the max tolerance.")
    return df.iloc[[]], tolerance  # Return an empty DataFrame

def find_oriented_bounding_rect(image, lower_color, upper_color):
    """    Detects the largest contour of a specified color in the image and returns its bounding rectangle.
    Args:
        image (numpy.ndarray): The input image in which to find the contour.
        lower_color (tuple): The lower bound of the color in HSV format.
        upper_color (tuple): The upper bound of the color in HSV format.
    Returns:
        tuple: A tuple containing the coordinates (x, y) and dimensions (width, height) of the bounding rectangle.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Combine masks to get only red pixels
    red_mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Threshold the mask to create a binary image
    _, thresh = cv2.threshold(red_mask, 127, 255, cv2.THRESH_BINARY)

    # Define the kernel for the morphological operation
    # ChatGPT gave me this part of the code, I have no idea how it works
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological opening to remove noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform morphological closing to close small holes
    # This really helps to cleanup the mask
    processed_img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        oriented_rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(oriented_rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 255, 255), 2)

    contour = max(contours, key=cv2.contourArea)
    
    # Get oriented bounding rectangle
    oriented_rect = cv2.minAreaRect(cont)
    box = cv2.boxPoints(oriented_rect)
    box = np.intp(box)

    (center_x, center_y), (width, height), angle = np.intp(oriented_rect)

    cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
    cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
    cv2.imshow("Frame", image)
    cv2.imshow("Mask", processed_img)
    cv2.waitKey(0)

    return (center_x, center_y, width, height, angle)
def find_oriented_matching_rows(df, target, start_tol=2, max_tol=15, step=2):
    """
    Find rows in the dataframe that match the target values within a dynamically
    increasing tolerance. If no rows are found for a very small tolerance, the
    tolerance will be increased until at least one row is found or the maximum
    tolerance is reached.

    Parameters:
        df (pd.DataFrame): DataFrame with the columns.
        target (dict): Target values for each column.
        start_tol (int): Starting tolerance measured in pixels.
        max_tol (int): Maximum allowed tolerance measured in pixels.
        step (int): Increment to increase tolerance on each iteration measured in pixels.

    Returns:
        filtered_df (pd.DataFrame): DataFrame with matching rows.
        used_tol (int): Tolerance at which the matching rows were found.
    """
    
    tolerance = start_tol
    while tolerance <= max_tol:
        # Create mask using np.isclose for each column
        mask = (
            np.isclose(df['Center_X'], target['Center_X'], atol=tolerance) &
            np.isclose(df['Center_Y'], target['Center_Y'], atol=tolerance) &
            np.isclose(df['Width'], target['Width'], atol=tolerance) &
            np.isclose(df['Height'], target['Height'], atol=tolerance) &
            np.isclose(df['Angle'], target['Angle'], atol=tolerance)
        )
        filtered_df = df[mask]
        
        # Check if any row is found
        if not filtered_df.empty:
            print(f"Found rows with tolerance: {tolerance}")
            return filtered_df, tolerance
        
        # Increase the tolerance and try again
        tolerance += step

    # No rows found within maximum tolerance
    print("No rows found within the max tolerance.")
    return df.iloc[[]], tolerance  # Return an empty DataFrame

def find_correct_df(image, df,target_res){
    #target res is like (1604,1300)
    #image is cv2 image of 1604,1300
    #df is dataframe with columns Center_X, Center_Y, Width, Height, Angle
    
}