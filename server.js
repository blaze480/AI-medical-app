const express = require("express");
const path = require("path");

const app = express();

const appFolder = path.join(__dirname, "app");

// Serve HTML, CSS, JS and other static files
app.use(express.static(appFolder));

app.get("/", (req, res) => {
    res.sendFile(path.join(appFolder, "index.html"));
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});