const $loginRegForm=document.getElementById("user-login")


$loginRegForm.addEventListener('submit',async (e)=>{
    e.preventDefault()
    const $email=document.getElementById("email").value
    const $password=document.getElementById("password").value
    console.log($email,$password)
    console.log("data captured");
    console.log($password.length)
    if($password.length > 0){
    const result = await fetch('/user/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            email:$email,
            password:$password
        })
    }).then((res) => res.json())
    console.log(result)
    if (!result.error) {
        sessionStorage.setItem("name",result.user.firstname)
        location.href="markets.html"
    } else {
        document.getElementById("password").classList.add("wrong")
        
        setTimeout(
          () => {
              document.getElementById("password").classList.remove("wrong")
              document.getElementById("message").innerHTML = "Password Incorrect";

          },1000
      )


    }
}
// else{
//     alert("Please enter all required fields")
// }


})