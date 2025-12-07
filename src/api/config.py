from pathlib import Path

# ENV when using standalone uvicorn server running FastAPI in api directory
ENV_PATH = Path('../../env/online.env')

DESCRIPTION = """
This API identifies patients at risk of developing breast cancer.\n
 
### Results 
**Cancer prediction:** *Maligant* if a patient will develop a breast cancer, and *Benign* otherwise\n

**Cancer probability:** In percentage\n

### Let's Connect
👨‍⚕️ `Gabriel Okundaye`\n
[<img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="20" height="20">  LinkendIn](https://www.linkedin.com/in/dr-gabriel-okundaye)

[<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="20" height="20">  GitHub](https://github.com/D0nG4667/sepsis_prediction_full_stack)
 
"""