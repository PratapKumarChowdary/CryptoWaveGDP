

const authName = sessionStorage.getItem("name")
 
if(!authName)
  location.href="login.html"
