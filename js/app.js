 // Get references to the video element and the upload button
 const videoElement = document.getElementById("video-element");
 const uploadButton = document.getElementById("upload-button");
 const uploadLabel = document.getElementById("upload-button-label");
 const cameraToggle = document.getElementById("camera-toggle");
 const darkModeToggle = document.getElementById("dark-mode-toggle");


 // Dark Mode
 darkModeToggle.addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
  
    // Toggle Dark Mode Text
    if (document.body.classList.contains("dark-mode")) {
      darkModeToggle.textContent = "Light Mode";
    } else {
      darkModeToggle.textContent = "Dark Mode";
    }
  });
  
  // Initial Text
  darkModeToggle.textContent = "Dark Mode";
  
  // Toggle Dark Mode Changes
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    document.body.classList.add("dark-mode");
    videoElement.classList.add("dark-mode");
    uploadLabel.classList.add("dark-mode");
    cameraToggle.classList.add("dark-mode");
    darkModeToggle.textContent = "Light Mode";
  }
 
 // Camera
 cameraToggle.addEventListener("click", () => {
   navigator.mediaDevices.getUserMedia({ video: true })
     .then(stream => {
       videoElement.srcObject = stream;
     })
     .catch(error => {
       console.error("Error accessing user camera:", error);
     });
 });

 // Upload
 uploadButton.addEventListener("change", () => {
   const file = uploadButton.files[0];
   const url = URL.createObjectURL(file);
   videoElement.src = url;
 });

 // TODO: Model
