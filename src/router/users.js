const users = require("../models/mongoSchema.js")
const express = require("express")
const router = new express.Router()


var savedOTPS = {

};
router.get("/test",(req,res)=>{
    res.send({
        name:"server is working"
    })
})

//endpoint for creating user
router.post("/user/signup",async(req,res)=>{
    console.log("Signup request recieved", req)
    const user = new users(req.body)
    console.log(user)
    try{
     await user.save()  
     console.log("email sent")
    //  res.cookie("Authorization",token)     
    //  res.cookie("type","student")     
     res.status(201).send({user})
    }catch(e){
        console.log(e)
     res.status(400).send({error:"unable to register"})
    }
  
 
 })

  //endpoint for login users
  router.post("/user/login",async(req,res)=>{
    try{
    const user = await users.findByCredentials(req.body.email, req.body.password)
    console.log(user)
    if(user){

  //  res.cookie("Authorization",token)
  //  res.cookie("type","student")     
    res.send({user})
    }
    else res.status(400).send({error:"Wrong password!"})
    }
    catch(e){
        console.log(e.message)
        res.status(400).send({error:e.message})
    }

 })

 module.exports = router