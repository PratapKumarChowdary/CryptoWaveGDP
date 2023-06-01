

logout = (e) =>{
    const confirmLogout=confirm("Are you sure you want to logout?");
    if(confirmLogout){
        sessionStorage.clear()
        location.href="login.html"
        
}
}