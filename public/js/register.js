const $userRegForm=document.getElementById("user-register")

$userRegForm.addEventListener('submit',async (e)=>{
    e.preventDefault()
    console.log("data validated")
    const $firstname=document.getElementById("firstname").value
    const $lastname=document.getElementById("lastname").value
    const $email=document.getElementById("email").value
    const $password=document.getElementById("password1").value
    const $confirmPassword=document.getElementById("password2").value
    console.log($password,$confirmPassword)
   function verifyPassword(password1,password2) {  
    value = password2
    if(password1!==password2){
        document.getElementById("message").innerHTML = "password and confirm password should be same";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
        return false
    }
    if(password1.length < 8){
        document.getElementById("message").innerHTML = "password length should be longer than or equal to 8 characters";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
        return false
    }
    const isNonWhiteSpace = /^\S*$/;
    if (!isNonWhiteSpace.test(value)) {
        document.getElementById("message").innerHTML ="Password must not contain Whitespaces.";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
      return false
    }
    const isContainsCharacter = /[a-zA-Z]/;
    if (!isContainsCharacter.test(value)) {
        document.getElementById("message").innerHTML ="Password must contain at leat one character.";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
      return false
    }
  
    const isContainsNumber = /^(?=.*[0-9]).*$/;
    if (!isContainsNumber.test(value)) {
        document.getElementById("message").innerHTML = "Password must contain at least one Digit.";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
        return false
    }
  
    const isContainsSymbol =
      /^(?=.*[~`!@#$%^&*()--+={}\[\]|\\:;"'<>,.?/_₹]).*$/;
    if (!isContainsSymbol.test(value)) {
        document.getElementById("message").innerHTML = "Password must contain at least one Special Symbol.";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
        return false
    }
  
    const isValidLength = /^.{8,16}$/;
    if (!isValidLength.test(value)) {
        document.getElementById("message").innerHTML = "Password must be 8-16 Characters Long.";
        setTimeout(
            () => {
                document.getElementById("message").innerHTML = " ";
            },4200
        )
        return false
    }
 
    return true
}
    if(verifyPassword($password,$confirmPassword)){
        
        console.log("Password same")
        const result = await fetch('/user/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                firstname:$firstname,
                lastname:$lastname,
                email:$email,
                password:$password
            })
        }).then((res) => res.json())
        console.log(result)
        if (!result.error) {
            alert("Successfully Registered")
            location.href="login.html"
        } else {
            alert(result.error)
        }
    }
    // else{
    //     alert("Please enter all required fields & password correctly")
    // }
})