/********************************************************************
* akfquiz5.js 
* for use with mkquiz 4.1.0
* Language: JavaScript 1.5
* Copyright (C) 2003-2005 Andreas K. Foerster <akfquiz@akfoerster.de>
*
* This file is part of AKFQuiz
*
* AKFQuiz is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* AKFQuiz is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
*********************************************************************/

/*
LIMITS:

- no graphic with checkboxes yet.
*/

  var Shown = false;  // answer shown already?
  
  var Empty = new Image();
  var Right = new Image();
  var Wrong = new Image();

  // preload the images
  Empty.src = "leer.png";
  Right.src = "richtig.png";
  Wrong.src = "falsch.png";

  function jumpAssessmentURI(Points, MaxPoints, Percent) {
     var URI = document.getElementById("assessmentURI").value;
     if( URI.charAt(URI.length-1) == "?" )
       URI = URI + "points=" + Points + 
                   "&maxpoints=" + MaxPoints +
		   "&percent=" + Percent;
     window.location.href = URI;
  }

  function assessmentPercent(Percent) {
     for(var i=0; i<asmntval.length; i++) {
       if(Percent >= asmntval[i]) { alert(asmnttxt[i]); break; }
     }
  }

  function DisableForm(disable) {
    var l = document.getElementsByTagName("input").length;
    for(var i=0;i<l;i++)
      with(document.getElementsByTagName("input")[i])
        if( type=="radio" || type=="checkbox" )
	  disabled=disable;
   }

  function ShowHints() { // make all hints visible
    var l = document.getElementsByTagName("div").length;
    var cl;
    
    for(var i=0; i<l; i++) {
      cl = document.getElementsByTagName("div")[i].className;
      if(cl=="hint" || cl=="assessment")
        document.getElementsByTagName("div")[i].style.display="block";
    }
  }

  function HideHints() { // make all hints invisible
    var l = document.getElementsByTagName("div").length;
    var cl;
    
    for(var i=0; i<l; i++) {
      cl = document.getElementsByTagName("div")[i].className;
      if(cl=="hint" || cl=="assessment")
        document.getElementsByTagName("div")[i].style.display="none";
    }
  }

  function Result(sol1, sol2, sol3, sol4, sol5, seen, MaxPoints) {

    function ShowImage(bildname,bild) {
       var l = document.getElementsByTagName("img").length;
       for(var b=0; b<l; b++)
         if(document.getElementsByTagName("img")[b].name==bildname) 
             document.getElementsByTagName("img")[b].src=bild.src;
       }

    function Evaluate() {
        var Points  = 0;
	var l = document.getElementsByTagName("input").length;

        for(var i=0; i<l; i++) {
           with(document.getElementsByTagName("input")[i]) {
             if(type=="radio" || type=="checkbox") {
	       // if checked, then evaluate
               if(checked) {
                 Points += eval(value);
                 // graphic just for radio buttons for now
                 if(type=="radio") {
                   if(eval(value)>0) { ShowImage(name, Right) }
                                else { ShowImage(name, Wrong) }
                      } } } } }

	ShowHints();
	
	var Percent = parseInt(Math.max(Points,0)*100/MaxPoints+.5);
        var Answer = sol1+Points+sol2+MaxPoints+sol3;
	if(Points>0)
            Answer += "\n"+sol4+Percent+"%."
          else
            Answer += "\n"+sol5;
        alert(Answer);
	
        // assessmentURI or assessmentPercent
	if(document.getElementById("assessmentURI"))
	    jumpAssessmentURI(Points, MaxPoints, Percent)
        else if(typeof(asmntval)!="undefined")
                assessmentPercent(Percent);
        }

  function SeenAlready() {
    alert(seen);
    }

  if(Shown) SeenAlready(); else Evaluate();
  }
  
  function markLabels() {
    var l = document.getElementsByTagName("label").length;
    for(var i=0; i<l; i++) {
      with(document.getElementsByTagName("label")[i]) {
        if(eval(document.getElementById(htmlFor).value)>0) 
          className="correct"
        else
          className="wrong" 
    } } }

  function unmarkLabels() {
    var l = document.getElementsByTagName("label").length;
    for(var i=0; i<l; i++) 
      document.getElementsByTagName("label")[i].className=""
    }

  function Solution(really) {
    var l;
    if(confirm(really)) {
      EraseImages();
      ShowHints();
      markLabels();
      l = document.getElementsByTagName("input").length;
      for(var i=0; i<l; i++) {
        with(document.getElementsByTagName("input")[i]) {
           if(type=="radio" || type=="checkbox") {
             if(eval(value)>0) checked = true; else checked = false;
 	     disabled=true; } } }
      location.href = "#top";
      Shown = true;
      } } 

  function EraseImages() {
     var l = document.getElementsByTagName("img").length;
     for(var i=0; i<l; i++)
        if(document.getElementsByTagName("img")[i].src==Right.src || 
           document.getElementsByTagName("img")[i].src==Wrong.src) 
             document.getElementsByTagName("img")[i].src=Empty.src;
     }

  function New() {
     Shown = false;
     EraseImages();
     HideHints();
     unmarkLabels();
     DisableForm(false);
     document.forms["akfquiz"].reset(); //@@@
     location.href = "#top";
     return true;
     }

