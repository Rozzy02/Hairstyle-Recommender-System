import os

import cv2
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from hairstyle_recommendation import (
    detect_face_shape,
    get_hairstyle_recommendations,
)


def webcam(request):
    """Display the webcam page."""
    return render(request, "webcam.html")


@csrf_exempt
def capture_and_recommend(request):
    """Save a captured image and request a prediction."""

    if request.method != "POST":
        return JsonResponse(
            {"error": "POST request required."},
            status=405
        )

    uploaded_image = request.FILES.get("image")

    if uploaded_image is None:
        return JsonResponse(
            {"error": "No image was uploaded."},
            status=400
        )

    if uploaded_image.size == 0:
        return JsonResponse(
            {"error": "The uploaded image is empty."},
            status=400
        )

    # Save the captured image inside Myapp
    save_directory = os.path.join(
        settings.BASE_DIR, "Myapp"
    )

    os.makedirs(save_directory, exist_ok=True)

    image_path = os.path.join(
        save_directory, "captured_photo.jpg"
    )

    try:
        # Decode the uploaded image
        image_bytes = uploaded_image.read()

        image_array = cv2.imdecode(
            __import__("numpy").frombuffer(
                image_bytes, dtype="uint8"
            ),
            cv2.IMREAD_COLOR
        )

        if image_array is None:
            return JsonResponse(
                {"error": "Invalid image file."},
                status=400
            )

        # Save as an actual JPG image
        saved = cv2.imwrite(image_path, image_array)

        if not saved:
            return JsonResponse(
                {"error": "Could not save the image."},
                status=500
            )

        # Predict using the saved image
        result = detect_face_shape(image_path)

        recommendations = get_hairstyle_recommendations(
            result["face_shape"]
        )

        return JsonResponse({
            "message": "Photo captured successfully.",
            "face_shape": result["face_shape"],
            "confidence": result["confidence"],
            "recommendations": recommendations,
        })

    except (ValueError, FileNotFoundError) as error:
        return JsonResponse(
            {"error": str(error)},
            status=400
        )

    except Exception:
        return JsonResponse(
            {"error": "An unexpected processing error occurred."},
            status=500
            )
