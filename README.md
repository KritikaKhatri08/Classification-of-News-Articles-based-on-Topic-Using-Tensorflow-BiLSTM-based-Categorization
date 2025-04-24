1. Open the project in VS Code 
Open vs code
Click file -> Open Folder
Select the folder that contains your project files
2. Check package. Json file
Open the file called package. Json in the project 
Make sure it includes this “scripts” section:
      "scripts": {
                       "dev": "Vite", 
                       "build": "Vite build", 
                       "preview": "Vite preview“
                     }
3. Install Dependencies
Open the terminal inside VS Code:
     npm install (this reads the package. Json and installs everything)
4. Run the App
Now in the same terminal, run:
     npm run dev
You will see:
     VITE v4.x.x  ready in xyz ms
     Local:   http://localhost:5173/
     Click the http://localhost:5173/   (link to view your app in the browser.)
5. Verify App Is Running
The browser opens
You see your React app running (e.g., “Hello Tailwind + React + TypeScript!”)

