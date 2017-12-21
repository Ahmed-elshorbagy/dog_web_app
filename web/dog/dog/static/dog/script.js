$.getJSON("http://127.0.0.1:8000/detect", function(data) {
    if (data.success) {
        $("#breed").text(data.breed);
        $("#con").attr("src", data.img);
    });
}
});