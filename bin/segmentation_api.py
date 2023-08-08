import requests
import json

# Import OpenCV2
import cv2


def segmentation_api(image_path):
    """Use an online API top segment an image into drivable space."""

    # Read the input image.
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Crop the image to a square.
    height, width = image.shape[:2]
    # image_size = min(height, width)
    # image = image[:image_size, :image_size, ...]
    
    # # Scale the image to the maximum size of 500 pixels.
    # image = cv2.resize(image, (500, 500))

    # Convert the image to the format expected by the API.
    image = image.tolist()
    image = [image]
    image = json.dumps(image)

    

    # Send the image to the API.
    headers = {'content-type': 'application/json'}
    response = requests.post(
        'https://api.mapbox.com/v4/segmentation/mapbox.driving/{0}/{1}?access_token={2}'.format(
            width, height, 'pk.eyJ1IjoidHNiZXJ0YWxhbiIsImEiOiJja3F2MmNrYWswYWRvMnZtbnpyenNjYXE5In0.PxsMKJEr_bVgiSEVrLlnrw'),
        data=image, headers=headers)

    print('Response:', response.status_code)
    if response.status_code != 200:
        print('Request failed:')
        print(response)

    # Convert the response to a dictionary.
    response_dict = json.loads(response.text)
    polygons = response_dict['features'][0]['geometry']['coordinates']
    # polygons = [polygon for polygon in polygons if len(polygon) > 2]

    # Return the polygons.
    return polygons

def main():

    image_dir = '/home/tsbertalan/Dropbox/data/gudrun/september_2/fisheye_2019-09-02-15-41-18_4.bag_images'

    import os
    image_path = os.path.join(image_dir, 'fisheye_2019-09-02-15-41-18_4-00177.png')

    polygons = segmentation_api(image_path)
    print(polygons)

    base_image = cv2.imread(image_path)


    # Draw the polygons.
    for polygon in polygons:
        for point in polygon:
            cv2.circle(base_image, tuple(point[0:2]), 3, (0, 0, 255), -1)

    # Display the image.
    cv2.imshow('image', base_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()